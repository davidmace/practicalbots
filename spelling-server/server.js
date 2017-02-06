var SpellCheck = require('hunspell-spellcheck');
    
var express = require('express');
var app = express();

// Escape a string so it can be used in regex parser
function escapeRegExp(str) {
  return str.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&");
}

// Change this if you need to support a different language
var spellcheck = new SpellCheck("en_US");

// Use:
// localhost:3002/?s=thankyou+dgo -> 'thank you dog'
app.get('/', function (req, res) {
  var text = req.query.s

  // Get hunspell's correction suggestions ie. {'thankyou':'thank you','dgo':'dog'}
  var suggestions = spellcheck.suggestion(text)

  for (var key in suggestions) {
    var answer = suggestions[key][0]

    // Find location of suggestion text and replace it
    // [^a-zA-Z\d] ensures we only match a complete word ie. not 'hat' in 'chatting'
    var reg = new RegExp('\\b' + escapeRegExp(key) + '\\b')
    var loc = text.match(reg).index
    text = text.substring(0, loc) + answer + text.substring(loc + key.length)
  }

  res.send(text)
});

app.listen(3002, function() {
  console.log('Spellchecker listening on port 3002!')
});
