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

function vq(obs, code_book) {
    if (!obs || !Array.isArray(obs) || obs.length === 0 || !code_book || !Array.isArray(code_book) || code_book.length === 0) {
        throw new Error('Invalid input. Both observation and code_book must be non-empty arrays.');
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

function encode(vecs, codewords, Ds, M, code_dtype) {
    console.log("encode ", vecs.length, codewords.length, Ds, M, code_dtype);
    // Check input requirements
    if (vecs.constructor !== Float32Array) {
        throw new Error("Input vectors must be of type Float32Array.");
    }
    if (vecs.length === 0 || vecs.length % (Ds * M) !== 0) {
        throw new Error("Input dimension must be Ds * M.");
    }
    const N = vecs.length / Ds;
    // Create an empty array to store the codes
    const codes = new code_dtype(M);
    // Loop through subspaces and encode
    for (let m = 0; m < M; m++) {
        const vecs_sub = vecs.subarray(m * Ds, (m + 1) * Ds);
        const result = vq(Array.from(vecs_sub), codewords[m]);
        codes[m] = result.index;
    }
    return codes;
}

// Function to compute indices of sorted distances between a query vector and a list of vectors
function computeSortedIndices(queryVector, vectorList) {
    const distancesWithIndices = [];
    for (let i = 0; i < vectorList.length; i++) {
        const vector = vectorList[i];
        try {
            const distance = euclideanDistance(queryVector, vector);
            distancesWithIndices.push({index: i, distance: distance});
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


async function loadBinaryFile(filePath, vectorSize) {
    try {
        const binaryData = fs.readFileSync(filePath);
        const vectors = [];
        for (let i = 0; i < binaryData.length; i += vectorSize) {
            const vector = binaryData.subarray(i, i + vectorSize);
            vectors.push(new Uint8Array(vector));
        }
        return vectors;
    } catch (error) {
        console.error("Error loading binary file: ${error.message}");
        return null;
    }
}


function readBinaryFileAndParseInt64s(filePath) {
  try {
    // Read the binary file synchronously.
    const data = fs.readFileSync(filePath);

    const int64Values = [];

    for (let i = 0; i < data.length; i += 8) {
      // Extract 8 bytes (64 bits) from the binary data.
      const binarySlice = data.slice(i, i + 8);

      // Convert the binary data to a BigInt.
      const int64Value = Number(BigInt('0x' + binarySlice.toString('hex')));
      
      // Push the BigInt value to the array.
      int64Values.push(int64Value);
    }

    return int64Values;
  } catch (error) {
    console.error('Error reading or parsing the binary file:', error);
    return [];
  }
}






async function loadIndices(filename) {
      const data = fs.readFileSync(filename);
  const integers = [];
  for (let i = 0; i < data.length / 8; i++) {
    integers.push(Number(data.slice(i * 8, (i + 1) * 8)));
  }
  return integers;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//
const conf = require('./conf.json');
const documents = require('./documents.json');
const indices = await readBinaryFileAndParseInt64s("indices.bin");
const vectors = await loadBinaryFile("pq.bin", conf['M']);
const codewords = require('./codewords.json');

let transformer = await import("@xenova/transformers");
const model_name = conf['model'].split('/')[1];
const model = 'Xenova/' + model_name;
let extractor = await transformer.pipeline('feature-extraction', model);

let query = "Audiovisual rights in sports events"; // 599320
console.log("Query: " + query);

let observation = await extractor(query, {pooling: "mean", normalize: true});
let result = encode(observation.data, codewords, conf['dim'] / conf['M'], conf['M'], Uint8Array);
console.log(result);

const sortedIndices = computeSortedIndices(result, vectors);
/**
for (let i = 0; i < 15; i++) {
    let row = documents[indices[sortedIndices[i]]];
    delete row.summary;
    console.log(i + " : " + row.reference + " - " + row.title);
}
**/

