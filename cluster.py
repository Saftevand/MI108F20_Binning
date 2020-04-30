import tensorflow as tf
import numpy as np
import random
from math import ceil
from collections import defaultdict as defaultdict, deque as deque

_DEFAULT_RADIUS = 0.06
# Distance within which to search for medoid point
_MEDOID_RADIUS = 0.05

_DELTA_X = 0.005
_XMAX = 0.3
#_XMAX = 0.6
# This is the PDF of normal with µ=0, s=0.01 from -0.075 to 0.075 with intervals
# of DELTA_X, for a total of 31 values. We multiply by _DELTA_X so the density
# of one point sums to approximately one
_NORMALPDF = _DELTA_X * tf.constant(
      [2.43432053e-11, 9.13472041e-10, 2.66955661e-08, 6.07588285e-07,
       1.07697600e-05, 1.48671951e-04, 1.59837411e-03, 1.33830226e-02,
       8.72682695e-02, 4.43184841e-01, 1.75283005e+00, 5.39909665e+00,
       1.29517596e+01, 2.41970725e+01, 3.52065327e+01, 3.98942280e+01,
       3.52065327e+01, 2.41970725e+01, 1.29517596e+01, 5.39909665e+00,
       1.75283005e+00, 4.43184841e-01, 8.72682695e-02, 1.33830226e-02,
       1.59837411e-03, 1.48671951e-04, 1.07697600e-05, 6.07588285e-07,
       2.66955661e-08, 9.13472041e-10, 2.43432053e-11])



class Cluster:
    __slots__ = ['medoid', 'seed', 'members', 'pvr', 'radius', 'isdefault', 'successes', 'attempts']

    def __init__(self, medoid, seed, members, pvr, radius, isdefault, successes, attempts):
        self.medoid = medoid
        self.seed = seed
        self.members = members
        self.pvr = pvr
        self.radius = radius
        self.isdefault = isdefault
        self.successes = successes
        self.attempts = attempts

    def __repr__(self):
        return '<Cluster of medoid {}, {} members>'.format(self.medoid, len(self.members))

    def as_tuple(self, labels):
        return labels[self.medoid], {labels[i] for i in self.members}

    def dump(self):
        return '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(self.medoid, self.seed, self.pvr, self.radius,
        self.isdefault, self.successes, self.attempts, ','.join([str(i) for i in self.members]))

    def __str__(self):
        radius = "{:.3f}".format(self.radius)
        if self.isdefault:
            radius += " (fallback)"

        return """Cluster of medoid {}
  N members: {}
  seed:      {}
  radius:    {}
  successes: {} / {}
  pvr:       {:.1f}
  """.format(self.medoid, len(self.members), self.seed, radius, self.successes, self.attempts, self.pvr)



class ClusterGenerator:
    __slots__ = ['MAXSTEPS', 'MINSUCCESSES', 'RNG', 'matrix', 'indices',
                 'seed', 'nclusters', 'peak_valley_ratio', 'attempts', 'successes',
                 'histogram', 'kept_mask']

    def __repr__(self):
        return "ClusterGenerator({} points, {} clusters)".format(len(self.matrix), self.nclusters)

    def __str__(self):
        return """ClusterGenerator({} points, {} clusters)
      MAXSTEPS:     {}
      MINSUCCESSES: {}
      pvr:          {}
      successes:    {}/{}
    """.format(len(self.matrix), self.nclusters, self.MAXSTEPS, self.MINSUCCESSES,
               self.peak_valley_ratio, self.successes, len(self.attempts))

    def __init__(self, matrix, maxsteps=25, windowsize=200, minsuccesses=20, destroy=False,
                normalized=False):

        #if not destroy:
        #    matrix = matrix.copy()

        # Shuffle matrix in unison to prevent seed sampling bias. Indices keeps
        # track of which points are which.
        normalized = normalize(matrix)
        normalized = tf.random.shuffle(normalized, 0)

        indices = tf.range(len(normalized))
        indices = tf.random.shuffle(indices, 0)


        self.MAXSTEPS = maxsteps
        self.MINSUCCESSES = minsuccesses
        self.RNG = random.Random(0)

        self.matrix = normalized
        # This refers to the indices of the original matrix. As we remove points, these
        # indices do not correspond to merely range(len(matrix)) anymore.
        self.indices = indices
        self.seed = -1
        self.nclusters = 0
        self.peak_valley_ratio = 0.1
        self.attempts = deque(maxlen=windowsize)
        self.successes = 0

        histogram, kept_mask = self.init_histogram_kept_mask(len(indices))
        self.histogram = tf.Variable(histogram)
        self.kept_mask = kept_mask

    def init_histogram_kept_mask(self, N):
        "N is number of contigs"

        kept_mask = tf.ones(N, dtype=tf.dtypes.bool)
        histogram = tf.zeros(ceil(_XMAX/_DELTA_X), dtype=tf.dtypes.double)

        return histogram, kept_mask

    def __iter__(self):
        return self

    def __next__(self):
        # Stop criterion. For CUDA, inplace masking the array is too slow, so the matrix is
        # unchanged. On CPU, we continually modify the matrix by removing rows.

        if not tf.reduce_any(self.kept_mask):
            raise StopIteration
        elif len(self.matrix) == 0:
            raise StopIteration

        cluster, medoid, points = self.findcluster()
        self.nclusters += 1
        updated_mask = self.kept_mask.numpy()
        for point in points.numpy():
            updated_mask[point] = False
        self.kept_mask = tf.convert_to_tensor(updated_mask)

        print("A cluster has been made")
        return cluster

    def findcluster(self):
        """Finds a cluster to output."""
        threshold = None

        # Keep looping until we find a cluster
        while threshold is None:
            # If on GPU, we need to take next seed which has not already been clusted out.
            # if not, clustered points have been removed, so we can just take next seed
            self.seed = (self.seed + 1) % len(self.matrix)
            while self.kept_mask[self.seed] == False:
                self.seed = (self.seed + 1) % len(self.matrix)

            medoid, distances = wander_medoid(self.matrix, self.kept_mask, self.seed, self.MAXSTEPS, self.RNG)

            # We need to make a histogram of only the unclustered distances - when run on GPU
            # these have not been removed and we must use the kept_mask

            histogram = tf.histogram_fixed_width(distances[self.kept_mask.numpy()], [0,_XMAX], len(self.histogram.numpy())).numpy()
            histogram[0] -= 1
            self.histogram = tf.convert_to_tensor(histogram)
            #self.histogram[0] -= 1 # Remove distance to self

            threshold, success = find_threshold(self.histogram, self.peak_valley_ratio)

            # If success is not None, either threshold detection failed or succeded.
            if success is not None:
                # Keep accurately track of successes if we exceed maxlen
                if len(self.attempts) == self.attempts.maxlen:
                    self.successes -= self.attempts.popleft()

                # Add the current success to count
                self.successes += success
                self.attempts.append(success)

                # If less than minsuccesses of the last maxlen attempts were successful,
                # we relax the clustering criteria and reset counting successes.
                if len(self.attempts) == self.attempts.maxlen and self.successes < self.MINSUCCESSES:
                    self.peak_valley_ratio += 0.1
                    self.attempts.clear()
                    self.successes = 0

        # These are the points of the final cluster AFTER establishing the threshold used
        points = smaller_indices(distances, self.kept_mask, threshold)
        points_within_threshold = tf.gather(params=self.indices, indices=points)
        isdefault = success is None and threshold == _DEFAULT_RADIUS and self.peak_valley_ratio > 0.55
        cluster = Cluster(self.indices[medoid], self.seed, points_within_threshold,
                        self.peak_valley_ratio,
                          threshold, isdefault, self.successes, len(self.attempts))
        return cluster, medoid, points


def sample_medoid(matrix, kept_mask, medoid, threshold):
    """Returns:
    - A vector of indices to points within threshold
    - A vector of distances to all points
    - The mean distance from medoid to the other points in the first vector
    """

    distances = calc_distances(matrix, medoid)
    cluster = smaller_indices(distances, kept_mask, threshold)

    if len(cluster) == 1:
        average_distance = 0.0
    else:

        average_distance = tf.reduce_sum(tf.gather(distances, cluster)) / (len(cluster) - 1)
        #average_distance = distances[cluster].sum().item() / (len(cluster) - 1)

    return cluster, distances, average_distance


def wander_medoid(matrix, kept_mask, medoid, max_attempts, rng):
    """Keeps sampling new points within the cluster until it has sampled
    max_attempts without getting a new set of cluster with lower average
    distance"""

    futile_attempts = 0
    tried = {medoid}  # keep track of already-tried medoids
    cluster, distances, average_distance = sample_medoid(matrix, kept_mask, medoid, _MEDOID_RADIUS)

    while len(cluster) - len(tried) > 0 and futile_attempts < max_attempts:
        sampled_medoid = rng.choice(cluster)

         # Prevent sampling same medoid multiple times.
        while sampled_medoid.numpy() in tried:
            sampled_medoid = rng.choice(cluster)

        tried.add(sampled_medoid.numpy())

        sampling = sample_medoid(matrix, kept_mask, sampled_medoid, _MEDOID_RADIUS)
        sample_cluster, sample_distances, sample_avg = sampling

        # If the mean distance of inner points of the sample is lower,
        # we move the medoid and reset the futile_attempts count
        if sample_avg < average_distance:
            medoid = sampled_medoid
            cluster = sample_cluster
            average_distance = sample_avg
            futile_attempts = 0
            tried = {medoid.numpy()}
            distances = sample_distances

        else:
            futile_attempts += 1

    return medoid, distances


def calc_distances(normalized, seed):
    seed_contig = normalized[seed]

    distances = 0.5 - tf.tensordot(normalized, seed_contig, axes=1)
    #distances[seed] = 0
    return distances


def calc_densities(histogram, pdf=_NORMALPDF):
    """Given an array of histogram, smoothes the histogram."""
    pdf_len = len(pdf)
    arr = np.zeros(len(histogram) + pdf_len - 1)
    for i in range(len(arr) - pdf_len + 1):
        arr[i:i + pdf_len] += pdf * histogram.numpy()[i]

    densities = tf.convert_to_tensor(arr[15:-15])

    return densities

def find_threshold(histogram, peak_valley_ratio):
    """Find a threshold distance, where where is a dip in point density
    that separates an initial peak in densities from the larger bulk around 0.5.
    Returns (threshold, success), where succes is False if no threshold could
    be found, True if a good threshold could be found, and None if the point is
    alone, or the threshold has been used.
    """
    peak_density = 0
    peak_over = False
    minimum_x = None
    density_at_minimum = None
    threshold = None
    success = False
    delta_x = _XMAX / len(histogram)

    # If the point is a loner, immediately return a threshold in where only
    # that point is contained.
    if tf.reduce_sum(histogram[:10]) == 0:
        return 0.025, None

    densities = calc_densities(histogram)

    # Else we analyze the point densities to find the valley
    x = 0
    for density in densities:
        # Define the first "peak" in point density. That's simply the max until
        # the peak is defined as being over.
        if not peak_over and density > peak_density:
            # Do not accept first peak to be after x = 0.1
            if x > 0.1:
                break
            peak_density = density

        # Peak is over when density drops below 60% of peak density
        if not peak_over and density < 0.6 * peak_density:
            peak_over = True
            density_at_minimum = density

        # If another peak is detected, we stop
        if peak_over and density > 1.5 * density_at_minimum:
            break

        # Now find the minimum after the peak
        if peak_over and density < density_at_minimum:
            minimum_x, density_at_minimum = x, density

            # If this minimum is below ratio * peak, it's accepted as threshold
            if density < peak_valley_ratio * peak_density:
                threshold = minimum_x
                success = True

        x += delta_x

    # Don't allow a threshold too high - this is relaxed with p_v_ratio
    if threshold is not None and threshold > 0.2 + peak_valley_ratio:
        threshold = None
        success = False

    # If ratio has been set to 0.6, we do not accept returning no threshold.
    if threshold is None and peak_valley_ratio > 0.55:
        threshold = _DEFAULT_RADIUS
        success = None

    return threshold, success


def smaller_indices(tensor, kept_mask, threshold):
    """Get all indices where the tensor is smaller than the threshold.
    Uses Numpy because Torch is slow - See https://github.com/pytorch/pytorch/pull/15190"""

    # If it's on GPU, we remove the already clustered points at this step.
    return tf.reshape(tensor=tf.where(((tensor <= threshold) & kept_mask)), shape=[-1])
    #return _torch.nonzero((tensor <= threshold) & kept_mask).flatten()

def normalize_pearson(contigs):

    row_means = tf.math.reduce_mean(contigs,1)
    means_column = tf.reshape(row_means, [-1, 1])
    mean_substracted = contigs - means_column
    norms = tf.norm(mean_substracted, axis=1)
    norms = tf.reshape(norms, [-1,1])
    normalized = mean_substracted / norms
    return normalized



def normalize(matrix, inplace=False):
    """Preprocess the matrix to make distance calculations faster.
    The distance functions in this module assumes input has been normalized
    and will not work otherwise.
    """

    #if not inplace:
    #    matrix = matrix.clone()

    # If any rows are kept all zeros, the distance function will return 0.5 to all points
    # inclusive itself, which can break the code in this module
    matrix = matrix / (tf.reshape(tf.norm(matrix, axis=1), shape=[-1,1]) * (2 ** 0.5))
    return matrix


def cluster(matrix, maxsteps=25, windowsize=200, minsuccesses=20, destroy=False,
            normalized=False):
    return ClusterGenerator(matrix, maxsteps, windowsize, minsuccesses, destroy, normalized)