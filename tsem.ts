
let vocabulary = new Array<string>();
let reverseVocabulary = new Map<string, number>();

let stoplist = new Set<string>("the and of to for in by with from on at or any i'm was when how a i it my you they this have is that but are".split(" "));
let wordPattern = /\w[\w\-\']*\w|\w/g;


const EULER_MASCHERONI = -0.5772156649015328606065121;
const PI_SQUARED_OVER_SIX = Math.PI * Math.PI / 6;
const HALF_LOG_TWO_PI = Math.log(2 * Math.PI) / 2;

const DIGAMMA_COEF_1 = 1/12;
const DIGAMMA_COEF_2 = 1/120;
const DIGAMMA_COEF_3 = 1/252;
const DIGAMMA_COEF_4 = 1/240;
const DIGAMMA_COEF_5 = 1/132;
const DIGAMMA_COEF_6 = 691/32760;
const DIGAMMA_COEF_7 = 1/12;
const DIGAMMA_COEF_8 = 3617/8160;
const DIGAMMA_COEF_9 = 43867/14364;
const DIGAMMA_COEF_10 = 174611/6600;

const DIGAMMA_LARGE = 9.5;
const DIGAMMA_SMALL = .000001;

/** Calculate digamma using an asymptotic expansion involving
Bernoulli numbers. */
function digamma(z: number): number {
    let psi = 0;
    
    if (z < DIGAMMA_SMALL) {
        psi = EULER_MASCHERONI - (1 / z); // + (PI_SQUARED_OVER_SIX * z);
        return psi;
    }

    while (z < DIGAMMA_LARGE) {
        psi -= 1 / z;
        z++;
    }

    let invZ = 1/z;
    let invZSquared = invZ * invZ;

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

function expDigamma(z: number) : number {
    if (z > 100) { return z - 0.5; }
    else { return Math.exp(digamma(z)); }
}

function getWordForID(wordID: number) {
    return vocabulary[wordID];
}

function getIDforWord(word: string) {
    if (reverseVocabulary.has(word)) {
        return reverseVocabulary.get(word);
    }
    let wordID = vocabulary.length;
    vocabulary.push(word);
    reverseVocabulary.set(word, wordID)
    return wordID;
}

interface Doc {
    tokens: Array<number>
}

let documents = Array<Doc>();

interface WordWeight {
    word: string,
    weight: number
}

type TopicWordWeights = Map<number, Array<WordWeight>>;

class Model {
    topicWordSmoothing: number;
    docTopicSmoothing: number;
    numTopics: number;
    topicWordScores: Float64Array;
    topicWordBuffer: Float64Array;
    docTopicWeights: Float64Array;
    docTopicBuffer: Float64Array;


    constructor(numTopics: number, topicWordSmoothing = 0.0, docTopicSmoothing = 0.0) {
        this.topicWordSmoothing = topicWordSmoothing;
        this.docTopicSmoothing = docTopicSmoothing;
        this.numTopics = numTopics;
        this.topicWordScores = new Float64Array(numTopics * vocabulary.length);
        this.topicWordBuffer = new Float64Array(numTopics * vocabulary.length);
        this.docTopicWeights = new Float64Array(this.numTopics);
        this.docTopicBuffer = new Float64Array(this.numTopics);

        for (let i = 0; i < this.topicWordBuffer.length; i++) {
            this.topicWordBuffer[i] = 2 + Math.random(); // mostly uniform
        }
        this.normalizeTopicWords();
    }

    iterate() {
        let tokenBuffer = new Float64Array(this.numTopics);
        let firstDoc = false;
        let totalLog = 0.0;
    
        for (let doc of documents) {
            this.docTopicWeights.fill(1.0 / this.numTopics);
            for (let iter = 0; iter < 5; iter++) {
                this.docTopicBuffer.fill(this.docTopicSmoothing);
                for (let token of doc.tokens) {
                    tokenBuffer.fill(0);
                    let sum = 0.0;
                    for (let topic = 0; topic < this.numTopics; topic++) {
                        tokenBuffer[topic] +=
                            this.topicWordScores[this.numTopics * token + topic] *
                            this.docTopicWeights[topic];
                        sum += tokenBuffer[topic];
                    }
    
                    let normalizer = 1.0 / sum;
                    for (let topic = 0; topic < this.numTopics; topic++) {
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
            for (let token of doc.tokens) {
                tokenBuffer.fill(0);
                let sum = 0.0;
                for (let topic = 0; topic < this.numTopics; topic++) {
                    tokenBuffer[topic] +=
                        this.topicWordScores[this.numTopics * token + topic] *
                        this.docTopicWeights[topic];
                    sum += tokenBuffer[topic];
                }
                totalLog += Math.log(sum);
    
                let normalizer = 1.0 / sum;
                for (let topic = 0; topic < this.numTopics; topic++) {
                    this.topicWordBuffer[this.numTopics * token + topic] +=
                        tokenBuffer[topic] * normalizer;
                }
            }
    
        }
        this.normalizeTopicWords();
    
        console.log("Total Log: " + totalLog);
    }

    normalizeDocTopicsDigamma(): void {
        let sum = 0.0;
        for (let topic = 0; topic < this.numTopics; topic++) {
            sum += this.docTopicBuffer[topic];
        }
        let normalizer = 1.0 / expDigamma(sum);
        for (let topic = 0; topic < this.numTopics; topic++) {
            this.docTopicWeights[topic] = expDigamma(this.docTopicBuffer[topic]) * normalizer;
        }
    }
    
    normalizeDocTopicsMAP(): void {
        let sum = 0.0;
        for (let topic = 0; topic < this.numTopics; topic++) {
            sum += this.docTopicBuffer[topic];
        }
        let normalizer = 1.0 / sum;
        for (let topic = 0; topic < this.numTopics; topic++) {
            this.docTopicWeights[topic] = this.docTopicBuffer[topic] * normalizer;
        }
    }

    normalizeDocTopics = this.normalizeDocTopicsMAP;

    normalizeTopicWordsDigamma(): void {
        let topicSums = new Float64Array(this.numTopics);
        for (let wordID = 0; wordID < vocabulary.length; wordID++) {
            for (let topic = 0; topic < this.numTopics; topic++) {
                topicSums[topic] += this.topicWordBuffer[this.numTopics * wordID + topic];
            }
        }

        let topicNormalizers = new Float64Array(this.numTopics);
        for (let topic = 0; topic < this.numTopics; topic++) {
            topicNormalizers[topic] = 1.0 / expDigamma(topicSums[topic]);
        }
        for (let wordID = 0; wordID < vocabulary.length; wordID++) {
            for (let topic = 0; topic < this.numTopics; topic++) {
                this.topicWordScores[this.numTopics * wordID + topic] = 
                    expDigamma(this.topicWordBuffer[this.numTopics * wordID + topic]) * topicNormalizers[topic];
            }
        }

        this.topicWordBuffer.fill(this.topicWordSmoothing);
    }

    normalizeTopicWordsMAP(): void {
        let topicSums = new Float64Array(this.numTopics);
        for (let wordID = 0; wordID < vocabulary.length; wordID++) {
            for (let topic = 0; topic < this.numTopics; topic++) {
                topicSums[topic] += this.topicWordBuffer[this.numTopics * wordID + topic];
            }
        }

        let topicNormalizers = new Float64Array(this.numTopics);
        for (let topic = 0; topic < this.numTopics; topic++) {
            topicNormalizers[topic] = 1.0 / topicSums[topic];
        }
        for (let wordID = 0; wordID < vocabulary.length; wordID++) {
            for (let topic = 0; topic < this.numTopics; topic++) {
                this.topicWordScores[this.numTopics * wordID + topic] = 
                    this.topicWordBuffer[this.numTopics * wordID + topic] * topicNormalizers[topic];
            }
        }

        this.topicWordBuffer.fill(this.topicWordSmoothing);
    }

    normalizeTopicWords = this.normalizeTopicWordsMAP;

    getTopicWords() : TopicWordWeights {
        let topicWordWeights = new Map<number, Array<WordWeight>>();
        for (let topic = 0; topic < this.numTopics; topic++) {
            topicWordWeights.set(topic, new Array<WordWeight>());
        }

        for (let wordID = 0; wordID < vocabulary.length; wordID++) {
            for (let topic = 0; topic < this.numTopics; topic++) {
                let wordWeight = { word: vocabulary[wordID],
                    weight: this.topicWordScores[this.numTopics * wordID + topic] };
                topicWordWeights.get(topic).push(wordWeight);
            }
        }

        for (let topic = 0; topic < this.numTopics; topic++) {
            let topicWeights = topicWordWeights.get(topic);
            topicWeights.sort((a, b) => b.weight - a.weight);
        }

        return topicWordWeights;
    }

    showTopicWords(limit = 10) {
        const formatter = new Intl.NumberFormat("en-US", {maximumFractionDigits: 2});
        let topicTopWords = this.getTopicWords();
        for (let topic = 0; topic < this.numTopics; topic++) {
            let topWords = topicTopWords.get(topic);
            console.log(topWords.slice(0,limit).map(
                d => d.word + " (" + formatter.format(d.weight) + ")"
            ).join(" "));
        }    
    }
}

/** Normalize an array to sum to one in place */
function sumToOne(x: Float64Array) : number {
    let sum = 0.0;
    for (let i = 0; i < x.length; i++) {
        sum += x[i];
    }
    let normalizer = 1.0 / sum;
    for (let i = 0; i < x.length; i++) {
        x[i] *= normalizer
    }
    return sum;
}

let wordCounts = new Map<string, number>();
let docCounts = new Map<string, number>();
function countDoc(tokens) {
    for (let s of tokens) {
        if (wordCounts.has(s)) {
            wordCounts.set(s, wordCounts.get(s) + 1);
        }
        else {
            wordCounts.set(s, 1);
        }
    }

    let wordSet = new Set<string>(tokens);
    for (let s of Array.from(wordSet.values())) {
        if (docCounts.has(s)) {
            docCounts.set(s, docCounts.get(s) + 1);
        }
        else {
            docCounts.set(s, 1);
        }
    }
}

function isValid(s) {
    return s.length > 0 && ! stoplist.has(s) && wordCounts.get(s) > 5 && docCounts.get(s) < 0.1 * totalDocs;
}
let totalDocs = 0;
let totalTokens = 0;

fetch("documents.txt")
.then(response => response.text())
.then(data => {
    // Pass through once to count
    for (let line of data.split("\n")) {
        let fields = line.split("\t");
        if (fields.length != 3) { continue; }
        let tokens = fields[2].toLocaleLowerCase().match(wordPattern).filter(s => ! stoplist.has(s) && s.length > 0);
        countDoc(tokens);
        totalDocs ++;
    }

    // And again to 
    for (let line of data.split("\n")) {
        let fields = line.split("\t");
        if (fields.length != 3) { continue; }
        let tokens = fields[2].toLocaleLowerCase().match(wordPattern).filter(isValid).map( getIDforWord );
        documents.push({"tokens": tokens});
        totalTokens += tokens.length;
    }
    console.log(documents.length);
});