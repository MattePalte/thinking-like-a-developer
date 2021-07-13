const http = require('http');
const express = require('express');
const path = require('path');
const yaml = require('js-yaml');
const fs = require('fs');
var crypto = require('crypto');

const bodyParser = require('body-parser');

const session = require('express-session');
const MongoDBStore = require('connect-mongodb-session')(session);

// Read configuration file
const config = yaml.load(fs.readFileSync('././config/settings.yaml', 'utf8'));
// prepare configuration options
let uri = config['mongodb_atlas']['endpoint'];
const username = config['mongodb_atlas']['username'];
const password = config['mongodb_atlas']['password'];
uri = uri.replace(/\<username\>/g, username).replace(/\<password\>/g, password);
const datasetFolder = config['dataset']['folder'];
const datasetFileName = config['dataset']['filename'];

const toolVersion = config['version'];
const nQuestionsPerExperiment = config['n_questions_per_experiment'];
const guidelineTask = config['guideline']['task_function_naming'];

const tools = require('./my_tools');
const database = require('./utils/database');
const port = process.env.PORT || 3000

const javascript_client = config['javascript_client'];

// Handy toolbox and helper functions
const app = express();
const sessionStore = new MongoDBStore({
    uri: uri,
    collection: 'sessions',
    expires: 1000 * 60 * 60 * 24 * 2  // 2 days
});


// Initialize session
app.use(
    session({
        secret: 'my-secret',
        resave: false,
        saveUninitialized: false,
        store: sessionStore
    })
);


// Register template engine
app.set('view engine', 'ejs');
app.set('views', 'views');


// Make static pages available together with css and javascript
app.use(express.static(path.join(__dirname, 'public')));


// BODY Parser functions
app.post('/result-receiver', bodyParser.json({ extended: true, limit: '10mb' }));

// handle submission of mouse traces logs and clicked tokens
app.post('/result-receiver', function (req, res, next) {
    console.log("post received: mouse trace");
    // save the log file use as name:
    // current time + id of the file

    let correctAnswers = 0;
    let wrongAnswers = 0;
    // Check how many correct asnwers
    if (req.session.correctAnswers) {
        correctAnswers = req.session.correctAnswers;
    }
    // Check how many wrong asnwers
    if (req.session.wrongAnswers) {
        wrongAnswers = req.session.wrongAnswers;
    }

    if ("id" in req.body && "mousetrace" in req.body) {
        // enrich the data with nickname before storing the human log
        req.body.nickname = req.session.nickname;
        req.body.randomcode = req.session.randomCode;
        req.body.experimentset = req.session.experiment_set;
        req.body.version = toolVersion;
        // check if the answer was right
        if (req.body.functionnamebyuser == req.session.expectedCorrectAnswer) {
            correctAnswers += 1;
            req.session.correctAnswers = correctAnswers;
        } else {
            wrongAnswers += 1;
            req.session.wrongAnswers = wrongAnswers;
        }
        //tools.saveHumanAttentionLog(fileId = req.body.id, content = 'Marco');
        tools.saveHumanAttentionLogToMongo(req.body);
    }
    console.log("Raw Body: ")
    console.log(req.body);
    // respond with a successful message
    res.setHeader('Content-Type', 'text');
    res.write("Saved Successfully.");
    res.end();
});


// READ THE NICKNAME FIELD

app.post("/login", bodyParser.urlencoded({ extended: true }));

app.post("/login", (req, res, next) => {
    const userInputNickName = req.body.nickname;
    console.log("Nickname: " + userInputNickName);
    const randomCodeNumber = Math.floor(Math.random() * 1000000000000000);
    const to_hash = userInputNickName + randomCodeNumber;
    let hashed_string = crypto.createHash('md5').update(to_hash).digest('hex');
    // md5 output 32 characters
    req.session.nickname = userInputNickName;
    req.session.randomCode = hashed_string;
    req.session.correctAnswers = 0;
    req.session.wrongAnswers = 0;
    res.redirect("/task-description");
});


// TASK DESCRIPTION PAGE
app.use("/task-description", (req, res, next) => {
    // Choose the experiment dataset to propose
    // Aka: ask MongoDB to have a list of available datasets
    // Upload available experiment sets to MongoDB
    fs.readdir(datasetFolder, function (err, files) {
        //handling error
        if (err) {
            return console.log('Unable to scan directory: ' + err);
        }
        let filtered_files = files.filter(function (file) {
            return file.startsWith('experiment_');
        });
        console.log('filtered_files')
        console.log(filtered_files)

        let minutesToBeReproposed = 120;
        database.mongoConnectDelivery(collection => {

            // we serve either brand new experiment sets
            // or those that have been delivered more than x minutes ago
            // and we still do not have a completion
            let cursor = collection.find(
                { $or: [
                    { delivered_to: 0 },
                    { $and: [
                        { delivered_to: { $gte: 1 }},
                        { completed_by: { $exists: false}},
                        { most_recent_time_when_served: { $lt: Date.now() - (1000 * 60 * minutesToBeReproposed)}}
                    ]}
                ]}
            );
            //let cursor = collection.find({ delivered_to: 0 });
            cursor.toArray( function(err, docs) {
                let servableDocuments = docs.map(doc => doc.filename);
                // let namesDelivered = docs.map(doc => doc.filename);
                // console.log('filtered_files: ' + filtered_files);
                // console.log('namesDelivered: ' + namesDelivered);
                // const neverDelivered =
                //     filtered_files.filter(file => !namesDelivered.includes(file));
                // if (neverDelivered.length == 0) {
                //     res.write('No more experiments set available, ask Matteo to deliver new ones.');
                //     res.end();
                // }
                // // Select one randomly
                // let selected_filename = neverDelivered[0];
                servableDocuments.sort();
                console.log(servableDocuments)
                let selected_filename = servableDocuments[0];
                if (!filtered_files.includes(selected_filename)) {
                    console.log('Unavailable selected document: ' + selected_filename);
                    res.write('Doc not found: ' + selected_filename);
                    res.end();
                }
                console.log(selected_filename);
                // Save the experiment filename as
                // session.delivered_dataset for the user
                req.session.experiment_set = selected_filename;
                console.log(selected_filename + ' delivered to ' + req.session.nickname);
                // update mongo (increment)
                // Tell MongoDB we are delivering it
                // Aka: create {'dataset': 'experiment_13', 'delivered': +1}
                collection.updateOne({ filename: selected_filename },
                { $inc: { delivered_to: 1 },
                  $max: { most_recent_time_when_served: Date.now()},
                  $push: { users: { by: req.session.nickname,
                                    code: req.session.randomCode,
                                    timestamp: Date.now()}
                        }
                });
                // serve the page
                console.log("Task description displayed...")
                res.render('task-description',{
                    toolVersion: toolVersion,
                    guideline: guidelineTask
                });
            })
        })
    });


});

// TASK DISCLAIMER PAGE
app.use("/task-disclaimer", (req, res, next) => {
    console.log("Task disclaimer displayed...");
    res.render('task-disclaimer',{
        toolVersion: toolVersion
    });
});

// TASK RATE PAGE

app.use("/rate-difficulty", bodyParser.urlencoded({ extended: true }));
app.post("/rate-difficulty", (req, res, next) => {
    console.log(req.body);
    if(req.body != 'undefined'
       && 'fileindex' in req.body
       && 'rating' in req.body) {
        const fileIndex = req.body.fileindex;
        const rating = req.body.rating;
        console.log('Received rating from: ' + req.session.randomCode +
                    ' - at position: ' + fileIndex +
                    ' - rating: ' + rating);
        req.body.nickname = req.session.nickname;
        req.body.randomcode = req.session.randomCode;
        req.body.experimentset = req.session.experiment_set;
        req.body.version = toolVersion;
        tools.saveRatingToMongo(req.body);
        if (req.session.nickname) {
            console.log('Next page...');
            const nextIndex = parseInt(fileIndex) + 1
            res.redirect("/experiment?positionOfFile=" + (parseInt(nextIndex)).toString());
            res.end();
            return;
        }
    }
    res.end();
});

app.use("/rate-difficulty", (req, res, next) => {

    // Check if the user inserted a user name
    if (!req.session.nickname) {
        res.redirect("/login");
        res.end();
        return;
    }

    let positionOfFile = 0;

    if (req.query != 'undefined' && 'positionOfFile' in req.query) {
        positionOfFile = parseInt(req.query.positionOfFile);
        console.log('Showing rater for file at position: ' + positionOfFile);
    }

    console.log("Rate question...");
    res.render('rate-difficulty',{
        toolVersion: toolVersion,
        positionOfFile: positionOfFile
    });
});



// START/LOGIN PAGE
app.use("/login", (req, res, next) => {
    console.log("Login displayed...")
    res.render('login', {
        toolVersion: toolVersion
    });
});


// END PAGE
app.use("/end-experiment", (req, res, next) => {
    console.log("End experiment...");
    let correctAnswers = 0;
    let wrongAnswers = 0;
    // Check how many correct answers
    if (req.session.correctAnswers) {
        correctAnswers = req.session.correctAnswers;
    }
    // Check how many wrong answers
    if (req.session.wrongAnswers) {
        wrongAnswers = req.session.wrongAnswers;
    }
    if ((correctAnswers + wrongAnswers) < nQuestionsPerExperiment) {
        res.write('Cheater? You have not completed 20 questions, go back and submit all of them.');
        res.end();
        return;
    }
    // inform the server that this experiment set has been Completed
    database.mongoConnectDelivery(collection => {
        // update mongo (increment) completed pages
        collection.updateOne({ filename: req.session.experiment_set },
            { $inc: { completed_by: 1 },
              $push: { completed_users: { by: req.session.nickname,
                                          timestamp: Date.now()}
                    }
            });
            // serve the page
            console.log("Experiment Completed: " + req.session.experiment_set);
            console.log("By: " + req.session.nickname);
            res.render('end-experiment-view', {
                randomCode: req.session.randomCode,
                correctAnswers: correctAnswers,
                wrongAnswers: wrongAnswers,
                totalAnswers: correctAnswers + wrongAnswers,
                toolVersion: toolVersion
            });
            res.end();
    })


});



app.use("/experiment", bodyParser.urlencoded({ extended: true }));

// Slideshow-view of code as tokenized (v02 of the App)
app.use("/experiment", (req, res, next) => {

    // Check if the user inserted a user name
    if (!req.session.nickname) {
        res.redirect("/login");
        res.end();
        return;
    }
    console.log("Nickname: " + req.session.nickname +
                " - RandomCode: " + req.session.randomCode);

    let correctAnswers = 0;
    let wrongAnswers = 0;
    // Check how many correct asnwers
    if (req.session.correctAnswers) {
        correctAnswers = req.session.correctAnswers;
    }
    // Check how many wrong asnwers
    if (req.session.wrongAnswers) {
        wrongAnswers = req.session.wrongAnswers;
    }

    let positionOfFile = 0;

    if (req.query != 'undefined' && 'positionOfFile' in req.query) {
        positionOfFile = parseInt(req.query.positionOfFile);
        console.log('Serving file at position: ' + positionOfFile);
    }
    const jsonFilePath = datasetFolder + req.session.experiment_set;

    var fs = require('fs');
    var allFunctionsData;
    fs.readFile(jsonFilePath, 'utf8', function (err, data) {
        if (err) throw err;
        // console.log(data);

        allFunctionsData = JSON.parse(data);
        fileInfo = allFunctionsData[positionOfFile];
        if (typeof fileInfo === 'undefined') {
            res.redirect('/end-experiment')
            //res.send("This index is not available");
            res.end();
            return;
        }
        // tokens = tools.prepareCodeBoxText(fileInfo.tokens, tokenClass='token');
        tokens = fileInfo.tokens_in_code
        console.log(tokens)


        let optionsFunctionNames = fileInfo.options;
        console.log(optionsFunctionNames);

        progressPercentage =
            parseInt((positionOfFile / allFunctionsData.length) * 100);


        //fileName: fileInfo.file_name,
        //functionName: fileInfo.function_name,
        req.session.expectedCorrectAnswer = fileInfo.option_correct;
        res.render('single-code-processed',
            {
                javascript_client: javascript_client,
                positionOfFile: positionOfFile,
                uuid: fileInfo.uuid,
                optionsFunctionNames: optionsFunctionNames,
                tokens: tokens,
                progress: progressPercentage,
                tokenClass: "token",
                nickname: req.session.nickname,
                correctAnswers: correctAnswers,
                wrongAnswers: wrongAnswers,
                totalAnswers: correctAnswers + wrongAnswers,
                guideline: guidelineTask
            });
    });
});

// HOME PAGE
app.use("/", (req, res, next) => {
    res.redirect('/login');
});


// Event Driven Architecture
const server = http.createServer(app);
server.listen(port);