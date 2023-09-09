#
# Copyright (c) 2023 Sebastien Campion.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

import json
import pickle
import sys

import nanopq
import numpy as np


def compute_pq(conf):
    embeddings_file = conf.get('embeddings', 'vectors.bin')
    dim = conf.get('dim', 384)
    M = conf.get('M', 8)
    print("Compute PQ with M ", M)
    X = np.frombuffer(open(embeddings_file, 'rb').read(), dtype="float32")
    X = np.reshape(X, (-1, dim))
    pq = nanopq.PQ(M=M)
    #pq = nanopq.OPQ(M=M)
    pq.fit(X)
    X_code = pq.encode(X)
    print("vectors ", X_code.shape)
    open("pq.bin", "wb").write(X_code.tobytes())
    with open("codewords.json", "w") as f:
        json.dump(pq.codewords.tolist(), f)
    with open('pq.pkl', 'wb') as f:
        pickle.dump(pq, f)


if __name__ == "__main__":
    conf = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    compute_pq(conf)

