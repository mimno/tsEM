var vocabulary = new Array();
var reverseVocabulary = new Map();
var stoplist = new Set("the and of to for in by with from on at or any i'm was when how a i it my you they this have is that but are".split(" "));
var wordPattern = /\w[\w\-\']*\w|\w/g;
var EULER_MASCHERONI = -0.5772156649015328606065121;
var PI_SQUARED_OVER_SIX = Math.PI * Math.PI / 6;
var HALF_LOG_TWO_PI = Math.log(2 * Math.PI) / 2;
var DIGAMMA_COEF_1 = 1 / 12;
var DIGAMMA_COEF_2 = 1 / 120;
var DIGAMMA_COEF_3 = 1 / 252;
var DIGAMMA_COEF_4 = 1 / 240;
var DIGAMMA_COEF_5 = 1 / 132;
var DIGAMMA_COEF_6 = 691 / 32760;
var DIGAMMA_COEF_7 = 1 / 12;
var DIGAMMA_COEF_8 = 3617 / 8160;
var DIGAMMA_COEF_9 = 43867 / 14364;
var DIGAMMA_COEF_10 = 174611 / 6600;
var DIGAMMA_LARGE = 9.5;
var DIGAMMA_SMALL = .000001;
/** Calculate digamma using an asymptotic expansion involving
Bernoulli numbers. */
function digamma(z) {
    var psi = 0;
    if (z < DIGAMMA_SMALL) {
        psi = EULER_MASCHERONI - (1 / z); // + (PI_SQUARED_OVER_SIX * z);
        return psi;
    }
    while (z < DIGAMMA_LARGE) {
        psi -= 1 / z;
        z++;
    }
    var invZ = 1 / z;
    var invZSquared = invZ * invZ;
    psi += Math.log(z) - .5 * invZ
        - invZSquared * (DIGAMMA_COEF_1 - invZSquared *
            (DIGAMMA_COEF_2 - invZSquared *
                (DIGAMMA_COEF_3 - invZSquared *
                    (DIGAMMA_COEF_4 - invZSquared *
                        (DIGAMMA_COEF_5 - invZSquared *
                            (DIGAMMA_COEF_6 - invZSquared *
                                DIGAMMA_COEF_7))))));
    return psi;
}
function expDigamma(z) {
    if (z > 100) {
        return z - 0.5;
    }
    else {
        return Math.exp(digamma(z));
    }
}
function getWordForID(wordID) {
    return vocabulary[wordID];
}
function getIDforWord(word) {
    if (reverseVocabulary.has(word)) {
        return reverseVocabulary.get(word);
    }
    var wordID = vocabulary.length;
    vocabulary.push(word);
    reverseVocabulary.set(word, wordID);
    return wordID;
}
var documents = Array();
var Model = /** @class */ (function () {
    function Model(numTopics, topicWordSmoothing, docTopicSmoothing) {
        if (topicWordSmoothing === void 0) { topicWordSmoothing = 0.0; }
        if (docTopicSmoothing === void 0) { docTopicSmoothing = 0.0; }
        this.normalizeDocTopics = this.normalizeDocTopicsMAP;
        this.normalizeTopicWords = this.normalizeTopicWordsMAP;
        this.topicWordSmoothing = topicWordSmoothing;
        this.docTopicSmoothing = docTopicSmoothing;
        this.numTopics = numTopics;
        this.topicWordScores = new Float64Array(numTopics * vocabulary.length);
        this.topicWordBuffer = new Float64Array(numTopics * vocabulary.length);
        this.docTopicWeights = new Float64Array(this.numTopics);
        this.docTopicBuffer = new Float64Array(this.numTopics);
        for (var i = 0; i < this.topicWordBuffer.length; i++) {
            this.topicWordBuffer[i] = 2 + Math.random(); // mostly uniform
        }
        this.normalizeTopicWords();
    }
    Model.prototype.iterate = function () {
        var tokenBuffer = new Float64Array(this.numTopics);
        var firstDoc = false;
        var totalLog = 0.0;
        for (var _i = 0, documents_1 = documents; _i < documents_1.length; _i++) {
            var doc = documents_1[_i];
            this.docTopicWeights.fill(1.0 / this.numTopics);
            for (var iter = 0; iter < 5; iter++) {
                this.docTopicBuffer.fill(this.docTopicSmoothing);
                for (var _a = 0, _b = doc.tokens; _a < _b.length; _a++) {
                    var token = _b[_a];
                    tokenBuffer.fill(0);
                    var sum = 0.0;
                    for (var topic = 0; topic < this.numTopics; topic++) {
                        tokenBuffer[topic] +=
                            this.topicWordScores[this.numTopics * token + topic] *
                                this.docTopicWeights[topic];
                        sum += tokenBuffer[topic];
                    }
                    var normalizer = 1.0 / sum;
                    for (var topic = 0; topic < this.numTopics; topic++) {
                        this.docTopicBuffer[topic] += tokenBuffer[topic] * normalizer;
                    }
                }
                this.normalizeDocTopics();
                if (firstDoc) {
                    console.log(this.docTopicWeights);
                }
            }
            firstDoc = false;
            // One more time to save word-topic weights
            for (var _c = 0, _d = doc.tokens; _c < _d.length; _c++) {
                var token = _d[_c];
                tokenBuffer.fill(0);
                var sum = 0.0;
                for (var topic = 0; topic < this.numTopics; topic++) {
                    tokenBuffer[topic] +=
                        this.topicWordScores[this.numTopics * token + topic] *
                            this.docTopicWeights[topic];
                    sum += tokenBuffer[topic];
                }
                totalLog += Math.log(sum);
                var normalizer = 1.0 / sum;
                for (var topic = 0; topic < this.numTopics; topic++) {
                    this.topicWordBuffer[this.numTopics * token + topic] +=
                        tokenBuffer[topic] * normalizer;
                }
            }
        }
        this.normalizeTopicWords();
        console.log("Total Log: " + totalLog);
    };
    Model.prototype.normalizeDocTopicsDigamma = function () {
        var sum = 0.0;
        for (var topic = 0; topic < this.numTopics; topic++) {
            sum += this.docTopicBuffer[topic];
        }
        var normalizer = 1.0 / expDigamma(sum);
        for (var topic = 0; topic < this.numTopics; topic++) {
            this.docTopicWeights[topic] = expDigamma(this.docTopicBuffer[topic]) * normalizer;
        }
    };
    Model.prototype.normalizeDocTopicsMAP = function () {
        var sum = 0.0;
        for (var topic = 0; topic < this.numTopics; topic++) {
            sum += this.docTopicBuffer[topic];
        }
        var normalizer = 1.0 / sum;
        for (var topic = 0; topic < this.numTopics; topic++) {
            this.docTopicWeights[topic] = this.docTopicBuffer[topic] * normalizer;
        }
    };
    Model.prototype.normalizeTopicWordsDigamma = function () {
        var topicSums = new Float64Array(this.numTopics);
        for (var wordID = 0; wordID < vocabulary.length; wordID++) {
            for (var topic = 0; topic < this.numTopics; topic++) {
                topicSums[topic] += this.topicWordBuffer[this.numTopics * wordID + topic];
            }
        }
        var topicNormalizers = new Float64Array(this.numTopics);
        for (var topic = 0; topic < this.numTopics; topic++) {
            topicNormalizers[topic] = 1.0 / expDigamma(topicSums[topic]);
        }
        for (var wordID = 0; wordID < vocabulary.length; wordID++) {
            for (var topic = 0; topic < this.numTopics; topic++) {
                this.topicWordScores[this.numTopics * wordID + topic] =
                    expDigamma(this.topicWordBuffer[this.numTopics * wordID + topic]) * topicNormalizers[topic];
            }
        }
        this.topicWordBuffer.fill(this.topicWordSmoothing);
    };
    Model.prototype.normalizeTopicWordsMAP = function () {
        var topicSums = new Float64Array(this.numTopics);
        for (var wordID = 0; wordID < vocabulary.length; wordID++) {
            for (var topic = 0; topic < this.numTopics; topic++) {
                topicSums[topic] += this.topicWordBuffer[this.numTopics * wordID + topic];
            }
        }
        var topicNormalizers = new Float64Array(this.numTopics);
        for (var topic = 0; topic < this.numTopics; topic++) {
            topicNormalizers[topic] = 1.0 / topicSums[topic];
        }
        for (var wordID = 0; wordID < vocabulary.length; wordID++) {
            for (var topic = 0; topic < this.numTopics; topic++) {
                this.topicWordScores[this.numTopics * wordID + topic] =
                    this.topicWordBuffer[this.numTopics * wordID + topic] * topicNormalizers[topic];
            }
        }
        this.topicWordBuffer.fill(this.topicWordSmoothing);
    };
    Model.prototype.getTopicWords = function () {
        var topicWordWeights = new Map();
        for (var topic = 0; topic < this.numTopics; topic++) {
            topicWordWeights.set(topic, new Array());
        }
        for (var wordID = 0; wordID < vocabulary.length; wordID++) {
            for (var topic = 0; topic < this.numTopics; topic++) {
                var wordWeight = { word: vocabulary[wordID],
                    weight: this.topicWordScores[this.numTopics * wordID + topic] };
                topicWordWeights.get(topic).push(wordWeight);
            }
        }
        for (var topic = 0; topic < this.numTopics; topic++) {
            var topicWeights = topicWordWeights.get(topic);
            topicWeights.sort(function (a, b) { return b.weight - a.weight; });
        }
        return topicWordWeights;
    };
    Model.prototype.showTopicWords = function (limit) {
        if (limit === void 0) { limit = 10; }
        var formatter = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
        var topicTopWords = this.getTopicWords();
        for (var topic = 0; topic < this.numTopics; topic++) {
            var topWords = topicTopWords.get(topic);
            console.log(topWords.slice(0, limit).map(function (d) { return d.word + " (" + formatter.format(d.weight) + ")"; }).join(" "));
        }
    };
    return Model;
}());
/** Normalize an array to sum to one in place */
function sumToOne(x) {
    var sum = 0.0;
    for (var i = 0; i < x.length; i++) {
        sum += x[i];
    }
    var normalizer = 1.0 / sum;
    for (var i = 0; i < x.length; i++) {
        x[i] *= normalizer;
    }
    return sum;
}
var wordCounts = new Map();
var docCounts = new Map();
function countDoc(tokens) {
    for (var _i = 0, tokens_1 = tokens; _i < tokens_1.length; _i++) {
        var s = tokens_1[_i];
        if (wordCounts.has(s)) {
            wordCounts.set(s, wordCounts.get(s) + 1);
        }
        else {
            wordCounts.set(s, 1);
        }
    }
    var wordSet = new Set(tokens);
    for (var _a = 0, _b = Array.from(wordSet.values()); _a < _b.length; _a++) {
        var s = _b[_a];
        if (docCounts.has(s)) {
            docCounts.set(s, docCounts.get(s) + 1);
        }
        else {
            docCounts.set(s, 1);
        }
    }
}
function isValid(s) {
    return s.length > 0 && !stoplist.has(s) && wordCounts.get(s) > 5 && docCounts.get(s) < 0.1 * totalDocs;
}
var totalDocs = 0;
var totalTokens = 0;
fetch("documents.txt")
    .then(function (response) { return response.text(); })
    .then(function (data) {
    // Pass through once to count
    for (var _i = 0, _a = data.split("\n"); _i < _a.length; _i++) {
        var line = _a[_i];
        var fields = line.split("\t");
        if (fields.length != 3) {
            continue;
        }
        var tokens = fields[2].toLocaleLowerCase().match(wordPattern).filter(function (s) { return !stoplist.has(s) && s.length > 0; });
        countDoc(tokens);
        totalDocs++;
    }
    // And again to 
    for (var _b = 0, _c = data.split("\n"); _b < _c.length; _b++) {
        var line = _c[_b];
        var fields = line.split("\t");
        if (fields.length != 3) {
            continue;
        }
        var tokens = fields[2].toLocaleLowerCase().match(wordPattern).filter(isValid).map(getIDforWord);
        documents.push({ "tokens": tokens });
        totalTokens += tokens.length;
    }
    console.log(documents.length);
});
