const MongoClient = require('mongodb').MongoClient;
const yaml = require('js-yaml');
const fs   = require('fs');

// Read configuration file
const config = yaml.load(fs.readFileSync('././config/settings.yaml', 'utf8'));
console.log(config);

// prepare configuration options
let uri = config['mongodb_atlas']['endpoint'];
const username = config['mongodb_atlas']['username'];
const password = config['mongodb_atlas']['password'];
uri = uri.replace(/\<username\>/g, username).replace(/\<password\>/g, password);
console.log(uri);
const databaseConfig = config['mongodb_atlas']['database'];
uri = uri.replace(/utils/g, databaseConfig);
console.log(uri)
const collectionConfig = config['mongodb_atlas']['collection'];
const experimentDeliveryConfig = config['mongodb_atlas']['collection_delivery'];
const experimentRatingConfig = config['mongodb_atlas']['collection_rating'];

// Launch the client
const client = new MongoClient(uri, { useUnifiedTopology: true });
let global_db = null;
client.connect().then((client) => {
  global_db = client.db(databaseConfig);
})

const mongoConnect = callback => {
  let collection = global_db.collection(collectionConfig);
  callback(collection);
};

const mongoConnectDelivery = callback => {
  let collection = global_db.collection(experimentDeliveryConfig);
  callback(collection);
};

const mongoConnectRating = callback => {
  let collection = global_db.collection(experimentRatingConfig);
  callback(collection);
};

module.exports = {
  mongoConnect,
  mongoConnectDelivery,
  mongoConnectRating
};