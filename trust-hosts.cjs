const hostile = require("hostile");

hostile.set("127.0.0.1", `${process.env.CN}`, function (err) {

  if (err) {

    console.error(err);
  } else {

    console.log("hosts successfully!");
  }
});

hostile.set("127.0.0.1", "localhost", function (err) {

  if (err) {

    console.error(err);
  } else {
    
    console.log("hosts successfully!");
  }
});