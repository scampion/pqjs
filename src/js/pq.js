let transformer = await import("@xenova/transformers")
let extractor = await transformer.pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');


function vq(obs, code_book) {
    if (!obs || !Array.isArray(obs) || obs.length === 0 || !code_book || !Array.isArray(code_book) || code_book.length === 0) {
        throw new Error('Invalid input. Both observation and code_book must be non-empty arrays.');
    }

    // Function to calculate the Euclidean distance between two vectors
    function euclideanDistance(vector1, vector2) {
        if (vector1.length !== vector2.length) {
            throw new Error('Vectors must have the same dimensionality for distance calculation.');
        }

        let sum = 0;
        for (let i = 0; i < vector1.length; i++) {
            sum += Math.pow(vector1[i] - vector2[i], 2);
        }

        return Math.sqrt(sum);
    }

    // Initialize variables to keep track of the closest codeword and its index
    let closestCodeWord = code_book[0];
    let closestIndex = 0;
    let minDistance = euclideanDistance(obs, closestCodeWord);

    // Iterate through the codebook to find the closest codeword
    for (let i = 1; i < code_book.length; i++) {
        const codeWord = code_book[i];
        const distance = euclideanDistance(obs, codeWord);

        if (distance < minDistance) {
            closestCodeWord = codeWord;
            closestIndex = i;
            minDistance = distance;
        }
    }

    // Return the closest codeword and its index
    return {
        codeword: closestCodeWord,
        index: closestIndex
    };
}


function encode(vecs, codewords, Ds, M,
                code_dtype) {
    // Check input requirements
    if (vecs.constructor !==  Float32Array) {
        throw new Error("Input vectors must be of type Float32Array.");
    }
    if (vecs.length === 0 || vecs.length % (Ds * M) !== 0) {
        throw new Error("Input dimension must be Ds * M.");
    }
    const N = vecs.length / Ds;
// Create an empty array to store the codes
    const codes = new code_dtype(N * M);
// Loop through subspaces and encode
    for (let m = 0; m < M; m++) {
        const vecs_sub = vecs.subarray(m * Ds, (m + 1) * Ds);
        const result = vq(Array.from(vecs_sub), codewords[m]);
        for (let n = 0; n < N; n++) {
            codes[n * M + m] = result.index;
        }
    }
    return codes;
}

async function loadBinaryFile(filePath) {
  try {
    const response = await fetch(filePath);
    if (!response.ok) {
      throw new Error(`Failed to fetch the file: ${response.statusText}`);
    }

    const buffer = await response.arrayBuffer();
    const uint8Array = new Uint8Array(buffer);

    // Assuming the file contains multiple 384-sized uint8 vectors
    const vectors = [];
    const vectorSize = 384;

    for (let i = 0; i < uint8Array.length; i += vectorSize) {
      const vector = uint8Array.slice(i, i + vectorSize);
      vectors.push(vector);
    }

    return vectors;
  } catch (error) {
    console.error(`Error loading binary file: ${error.message}`);
    return null;
  }
}

const codewords = require('./codewords.json');

let query = "Audiovisual rights in sports events"; // 599320
let observation = await extractor(query, {pooling: "mean", normalize: true});
let result = encode(observation.data, codewords, 96, 4, Uint8Array);
console.log(result);




const filePath = 'X_code_4.bin';
const fs = require("fs");

function loadBinaryFile(filePath) {
  try {
    const binaryData = fs.readFileSync(filePath);

    // Assuming the file contains multiple 384-sized uint8 vectors
    const vectors = [];
    //const vectorSize = 384;
    const vectorSize = 16;

    for (let i = 0; i < binaryData.length; i += vectorSize) {
      const vector = binaryData.slice(i, i + vectorSize);
      vectors.push(new Uint8Array(vector)); // Cast to Uint8Array
    }

    return vectors;
  } catch (error) {
    console.error(`Error loading binary file: ${error.message}`);
    return null;
  }
}

const vectors = loadBinaryFile(filePath);

// Function to compute distances between a query vector and a list of vectors
function computeDistances(queryVector, vectorList) {
  const distances = [];
  for (const vector of vectorList) {
    const distance = euclideanDistance(queryVector, vector);
    distances.push(distance);
  }
  return distances;
}



function euclideanDistance(vector1, vector2) {
  if (vector1.length !== vector2.length) {
    throw new Error("Vector dimensions do not match " + vector1.length + " != " + vector2.length);
  }

  let sum = 0;
  for (let i = 0; i < vector1.length; i++) {
    const diff = vector1[i] - vector2[i];
    sum += diff * diff;
  }

  return Math.sqrt(sum);
}

// Function to compute indices of sorted distances between a query vector and a list of vectors
function computeSortedIndices(queryVector, vectorList) {
  const distancesWithIndices = [];
  for (let i = 0; i < vectorList.length; i++) {
    const vector = vectorList[i];
    try {
        const distance = euclideanDistance(queryVector, vector);
        distancesWithIndices.push({ index: i, distance: distance });
    } catch (error) {
        console.error(`Error computing distance for vector ${i}: ${error.message} ` + vector.length + " != " + queryVector.length);
    }
  }

  // Sort the distancesWithIndices array based on distances
  distancesWithIndices.sort((a, b) => a.distance - b.distance);

  // Extract and return the sorted indices
  const sortedIndices = distancesWithIndices.map((item) => item.index);
  return sortedIndices;
}




const sortedIndices = computeSortedIndices(result, vectors);
//console.log("Sorted Indices:", sortedIndices.subarray(0, 50));


let documents;
fs.readFile('documents.txt', 'utf8', (err, data) => {
    if (err) {
        console.error(err);
        return;
    }
    // Split the file contents into an array of lines
    documents = data.split('\n');
});


const indData = fs.readFileSync("index.bin");
// Create a DataView to read the binary data as uint16 values
const dataView = new DataView(indData.buffer);
const uint16Array = [];
for (let i = 0; i < dataView.byteLength; i += 2) {
    const uint16Value = dataView.getUint16(i, true); // 'true' for little-endian format
    uint16Array.push(uint16Value);
}

const indices = new Uint16Array(uint16Array);

for(let i = 0; i < 60; i++){
    console.log(i + " : " + documents[indices[sortedIndices[i]]]);
}