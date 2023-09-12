/*
@licstart  The following is the entire license notice for the
JavaScript code in this page.

Copyright (C) 2014  Sebastien Campion

The JavaScript code in this page is free software: you can
redistribute it and/or modify it under the terms of the GNU
General Public License (GNU GPL) as published by the Free Software
Foundation, either version 3 of the License, or (at your option)
any later version.  The code is distributed WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU GPL for more details.

As additional permission under GNU GPL version 3 section 7, you
may distribute non-source (e.g., minimized or compacted) forms of
that code without the copy of the GNU GPL normally required by
section 4, provided you include this license notice and a URL
through which recipients can access the Corresponding Source.


@licend  The above is the entire license notice
for the JavaScript code in this page.
*/

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
        minDistance: minDistance,
        index: closestIndex
    };
}

function  encode(vecs, codewords, Ds, M, code_dtype) {
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

function binarySearch(arr, target) {
  let left = 0;
  let right = arr.length - 1;

  while (left <= right) {
    let mid = Math.floor((left + right) / 2);

    if (arr[mid] === target) {
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  return left;
}

// Function to compute indices of sorted distances between a query vector and a list of vectors
function computeSortedIndicesByDistance(queryVector, vectorList) {
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
    //const sortedIndices = distancesWithIndices.map((item) => item.index);
    //return sortedIndices;
    return distancesWithIndices;
}

function argsort(array) {
  // Create an array of indices [0, 1, 2, ...] for the input array
  const indices = Array.from(array.keys());
  // Sort the indices based on the values in the input array
  indices.sort((a, b) => array[a] - array[b]);
  return indices;
}





async function loadBinaryFile(filePath, vectorSize) {
    try {
        // Read the binary file synchronously.
        const response = await fetch(filePath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const binaryData = await response.arrayBuffer();
        const vectors = [];
        for (let i = 0; i < binaryData.byteLength; i += vectorSize) {
            const vector = binaryData.slice(i, i + vectorSize);
            vectors.push(new Uint8Array(vector));
        }
        return vectors;
    } catch (error) {
        console.error("Error loading binary file: ", error);
        return null;
    }
}


// Function to compute indices of sorted distances between a query vector and a list of vectors
function get_indices(documents){
    const indices = documents.map(doc => doc.nb_of_embeddings);
    // Calculate cumulative sums
    for (let i = 1; i < indices.length; i++) {
      indices[i] += indices[i - 1];
    }
    return indices;
}




function feature_position_to_doc_id(sortedIndices, indices, max_results) {
    const doc_ids_results = {};
    //const sortedIndices = distancesWithIndices.map((item) => item.index); // use dietance in a future version
    let nb_of_feats = 0;
    for (let i = 0; i < sortedIndices.length; i++) {
        const doc_i = binarySearch(indices, sortedIndices[i]);
        doc_ids_results[doc_i] = (doc_ids_results[doc_i] || 0) + 1;
        nb_of_feats += 1;
        if (Object.keys(doc_ids_results).length > max_results && nb_of_feats > max_results*max_results ) {
            break;
        }
    }
    return doc_ids_results;
}


function dtable(query, codewords, Ds, M, Ks) {
  if (Ks === undefined) {
      throw new Error('Ks value must be defined');
  }
  const D = query.length;
  if (D !== Ds * M) {
    throw new Error('Input dimension must be Ds * M');
  }
  // Create an empty distance table
  const dtable = new Array(M);
  for (let m = 0; m < M; m++) {
    dtable[m] = new Float32Array(Ks);
  }
  // Calculate distances
  for (let m = 0; m < M; m++) {
    const querySub = query.subarray(m * Ds, (m + 1) * Ds);
    for (let ks = 0; ks < Ks; ks++) {
      // Replace metric_function_map with the appropriate metric function
      dtable[m][ks] = euclideanDistance(querySub, codewords[m][ks]);
    }
  }
  return dtable;
}

function adist(codes, dtable) {
  // Check input dimensions
  if (codes.length === 0 || codes[0].length !== dtable.length) {
    throw new Error('Invalid input dimensions');
  }
  const N = codes.length;
  // Compute Asymmetric Distances
  const dists = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    dists[n] = 0;
    for (let m = 0; m < dtable.length; m++) {
      dists[n] += dtable[m][codes[n][m]];
    }
  }
  return dists;
}

function _search(dists, indices, documents, k_max=100){
    const doc_counter = {};

    // Calculate the indices of the sorted dists array
    const sortedIndices = dists.map((_, i) => i).sort((a, b) => dists[a] - dists[b]);

    console.log("")

    // Iterate over the first k_max sorted indices
    for (let i = 0; i < k_max && i < sortedIndices.length; i++) {
        const doc_i = binarySearch(indices, sortedIndices[i]);
        if (!doc_counter[doc_i]) {
            doc_counter[doc_i] = [];
        }
        doc_counter[doc_i].push(dists[sortedIndices[i]]);
    }

    const sortedDocCounter = Object.entries(doc_counter).sort((a, b) => {
        const meanA = a[1].reduce((acc, val) => acc + val, 0) / a[1].length;
        const meanB = b[1].reduce((acc, val) => acc + val, 0) / b[1].length;
        return meanA - meanB;
    });

    const results = [];
    for (let i = 0; i < sortedDocCounter.length; i++) {
        const doc = documents[sortedDocCounter[i][0]];
        results.push(doc);
    }
    return results;
}

function search(documents, query, codewords, vectors, conf, max_results){
    const indices = get_indices(documents);
    //let query_q = encode(query, codewords, conf['dim'] / conf['M'], conf['M'], Uint8Array);
    const dist_table = dtable(query, codewords,conf['dim'] / conf['M'], conf['M'], conf['Ks']);
    const distances = adist(vectors, dist_table);
    const results =  _search(distances, indices, documents, 100);
    return results;

}

export { search, loadBinaryFile };
