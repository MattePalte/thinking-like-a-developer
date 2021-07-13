// my_tools.js
// ========

const fs = require('fs');
const path = require('path');
const database = require('./utils/database');

class Token {
  constructor(uid, id4nlp, id4identifier, line, text, classname, charStart=0) {
      // https://stackoverflow.com/questions/192048/can-an-html-element-have-multiple-ids
      // <p id="foo" xml:id="bar">
      this.uid = uid; // always incremental and unique (given also to useless space tokens)
      this.id4nlp = id4nlp; // id that separate natural language chunks of the same identifiers
      this.id4identifier = id4identifier; // id that treat a variable as single token
      this.line = line; // id related to the line
      this.charStart = charStart; // position of the first start
      this.text = text; // textual content of the token
      this.classname = classname; // class of the token
  }
}


module.exports = {
    /**
     * Save the mouse trace (movements) as a file.
     *
     * @param fileID id of the file
     * @param content content of the file
     * @param baseFolder path where to save your file
     * @returns
     */
    saveHumanAttentionLog: function (fileId, content, baseFolder='./data/human_attention_log') {
      // create file name
      var d = new Date();
      var currentTimeEvent = d.getTime();
      var fileName = fileId + "-" + currentTimeEvent + '.json';
      // save file
      fs.writeFile(path.join(baseFolder, fileName), content, function (err) {
        if (err) return console.log(err);
        console.log('Successfully saved');
      });
    },
    /**
     * Prepare tokens to be displayed.
     *
     * @param tokens list raw tokens
     * @param tokenClass default class given to all code token
     * @returns objTokens list augmented tokens with all info to be printed
     */
    prepareCodeBoxText: function (tokens, tokenClass) {
      const objTokens = [];
      let uid = 0;
      let id4nlp = 0;
      let id4identifier = 0;
      let sameIdentifier = false;
      let line = 0;
      // true if we are within two <id></id> tokens
      for(let i = 0; i < tokens.length; i++) {
        t = tokens[i];
        if (t == "<id>") {
          id4identifier++;
          sameIdentifier = true;
        } else if (t == "</id>") {
          id4identifier++;
          sameIdentifier = false;
        } else {
          // if not a "meta" token then...
          // add token as it is
          tObj = new Token(uid, id4nlp, id4identifier, line, t, tokenClass);
          objTokens.push(tObj);
          // increment for the next
          uid++;
          id4nlp++;
          if (sameIdentifier == false) id4identifier++;
        }
      }
      const spacedObjToken = [];
      const punctList = ['[',']','{','}','(',')','!',':',',', '.', ';'];
      const punctNewLineList = ['{','}',';'];
      // add the first sentence start token by defaults
      spacedObjToken.push(objTokens[0])

      line = 0;
      // iteration for spaces and new lines
      for(let i = 0; i < (objTokens.length - 1); i++) {
        let currentToken = objTokens[i];
        let nextToken = objTokens[i + 1];
        if (currentToken['id4identifier'] == nextToken['id4identifier']) {
          // transform upper case first letter
          nextToken['text'] = nextToken['text'].charAt(0).toUpperCase() + nextToken['text'].slice(1);
          nextToken['line'] = line;
          // add underscore if same identifier
          spacedObjToken.push(nextToken);
        } else if (!punctList.includes(currentToken['text']) && !punctList.includes(nextToken['text'])) {
          // add empty space
          // console.log(">" + currentToken['text'] + "< add space >" + nextToken['text'] + "<");
          tObj = new Token("", "", "", line, " ", "t");
          spacedObjToken.push(tObj);
          nextToken['line'] = line;
          spacedObjToken.push(nextToken);
        } else if (punctNewLineList.includes(currentToken['text'])) {
          // add new line
          tObj = new Token("", "", "", line,  "\n", "t");
          spacedObjToken.push(tObj);
          // increase line counting
          line++;
          nextToken['line'] = line;
          spacedObjToken.push(nextToken);
        } else {
          nextToken['line'] = line;
          spacedObjToken.push(nextToken);
        }
      }

      let charStart = 0;
      let oldLine = 0;
      // Add the charStart for every token
      for (i = 0; i < spacedObjToken.length; i++) {
        let t = spacedObjToken[i];
        if (t['line'] == oldLine) {
          t['charStart'] = charStart;
        } else {
          // reset if new line starts
          t['charStart'] = 0;
          charStart = 0;
        }
        let len = t['text'].toString().length;
        charStart += parseInt(len);
        oldLine = t['line'];
      }

      return spacedObjToken;
      // return tokens.join(" ").replace(/[\{] /g, '{\n').replace(/[}] /g, '}\n').replace(/[;] /g, ';\n');
    },
    /**
    * Save the session as a document in mongodb.
    *
    * @param obj4Mongo object to store in the remote mongodb istance
    */
   saveHumanAttentionLogToMongo: function (obj4Mongo) {
     console.log("Send To MongoDB");
     var d = new Date();
     var currentTimeEvent = d.getTime();
     obj4Mongo["time"] = parseInt(currentTimeEvent);
     database.mongoConnect(collection => {
       console.log("run mongo connection");
       collection.insertOne(obj4Mongo, function(err, res) {
         if (err) throw err;
         console.log("1 document inserted");
         //console.log(obj4Mongo);
       });
       //console.log(collection);
     });

   },
   /**
   * Save the rating as a document in mongodb.
   *
   * @param rating4Mongo object to store in the remote mongodb instance
   */
  saveRatingToMongo: function (rating4Mongo) {
    console.log("Send rating to MongoDB");
    var d = new Date();
    var currentTimeEvent = d.getTime();
    rating4Mongo["time"] = parseInt(currentTimeEvent);
    database.mongoConnectRating(collection => {
      //console.log("run mongo connection");
      collection.insertOne(rating4Mongo, function(err, res) {
        if (err) throw err;
        console.log("1 rating inserted");
      });
    });

  },
   camelize: function (str) {
      return str.replace(/(?:^\w|[A-Z]|\b\w)/g, function(word, index) {
        return index === 0 ? word.toLowerCase() : word.toUpperCase();
      }).replace(/\s+/g, '');
   }
  };