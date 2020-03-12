#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import sys as _sys
import numpy as _np
import gzip as _gzip
import bz2 as _bz2
import lzma as _lzma
import os as _os
import multiprocessing as _multiprocessing
import time as _time
from hashlib import md5 as _md5
import bamnostic as _pysam
#import pysam as _pysam
#import vamb.vambtools as _vambtools
import array
#from cpython cimport array

class Reader:
    """Use this instead of `open` to open files which are either plain text,
    gzipped, bzip2'd or zipped with LZMA.

    Usage:
    >>> with Reader(file, readmode) as file: # by default textmode
    >>>     print(next(file))
    TEST LINE
    """

    def __init__(self, filename, readmode='r'):
        if readmode not in ('r', 'rt', 'rb'):
            raise ValueError("the Reader cannot write, set mode to 'r' or 'rb'")
        if readmode == 'r':
            self.readmode = 'rt'
        else:
            self.readmode = readmode

        self.filename = filename

        with open(self.filename, 'rb') as f:
            signature = f.peek(8)[:8]

        # Gzipped files begin with the two bytes 0x1F8B
        if tuple(signature[:2]) == (0x1F, 0x8B):
            self.filehandle = _gzip.open(self.filename, self.readmode)

        # bzip2 files begin with the signature BZ
        elif signature[:2] == b'BZ':
            self.filehandle = _bz2.open(self.filename, self.readmode)

        # .XZ files begins with 0xFD377A585A0000
        elif tuple(signature[:7]) == (0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00):
            self.filehandle = _lzma.open(self.filename, self.readmode)

        # Else we assume it's a text file.
        else:
            self.filehandle = open(self.filename, self.readmode)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.filehandle.close()

    def __iter__(self):
        return self.filehandle

def read_contigs(filehandle, minlength=100, preallocate=False):
    if minlength < 4:
        raise ValueError('Minlength must be at least 4, not {}'.format(minlength))

    if preallocate:
        print("Warning: Argument 'preallocate' in function read_contigs is deprecated."
        " read_contigs is now always memory efficient.",
        file=_sys.stderr)

    tnfs = PushArray(_np.float32)
    #lengths = PushArray(_np.int)
    #contignames = list()

    entries = byte_iterfasta(filehandle)

    for entry in entries:
        if len(entry) < minlength:
            continue

        tnfs.extend(entry.fourmer_freq())
        #lengths.append(len(entry))
        #contignames.append(entry.header)

    tnfs_arr = tnfs.take().reshape(-1, 136)
    #lengths_arr = lengths.take()

    return tnfs_arr#, contignames, lengths_arr

class PushArray:
    __slots__ = ['data', 'capacity', 'length']

    def __init__(self, dtype, start_capacity=1<<16):
        self.capacity = start_capacity
        self.data = _np.empty(self.capacity, dtype=dtype)
        self.length = 0

    def _grow(self, mingrowth):
        growth = max(int(self.capacity * 0.125), mingrowth)
        nextpow2 = 1 << (growth - 1).bit_length()
        self.capacity = self.capacity + nextpow2
        self.data.resize(self.capacity, refcheck=False)

    def append(self, value):
        if self.length == self.capacity:
            self._grow(64)

        self.data[self.length] = value
        self.length += 1

    def extend(self, values):
        lenv = len(values)
        if self.length + lenv > self.capacity:
            self._grow(lenv)

        self.data[self.length:self.length+lenv] = values
        self.length += lenv

    def take(self):
        "Return the underlying array"
        self.data.resize(self.length, refcheck=False)
        self.capacity = self.length
        return self.data

class FastaEntry:
    """One single FASTA entry. Instantiate with string header and bytearray
    sequence."""

    basemask = bytearray.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV',
                               b'ACGTTTNNNNNNNNNNNNNNNNNNNNN')
    __slots__ = ['header', 'sequence']

    def __init__(self, header, sequence):
        if len(header) > 0 and (header[0] in ('>', '#') or header[0].isspace()):
            raise ValueError('Header cannot begin with #, > or whitespace')
        if '\t' in header:
            raise ValueError('Header cannot contain a tab')

        masked = sequence.translate(self.basemask, b' \t\n\r')
        stripped = masked.translate(None, b'ACGTN')
        if len(stripped) > 0:
            bad_character = chr(stripped[0])
            msg = "Non-IUPAC DNA byte in sequence {}: '{}'"
            raise ValueError(msg.format(header, bad_character))

        self.header = header
        self.sequence = masked

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return '>{}\n{}'.format(self.header, self.sequence.decode())

    def format(self, width=60):
        sixtymers = range(0, len(self.sequence), width)
        spacedseq = '\n'.join([self.sequence[i: i+width].decode() for i in sixtymers])
        return '>{}\n{}'.format(self.header, spacedseq)

    def __getitem__(self, index):
        return self.sequence[index]

    def __repr__(self):
        return '<FastaEntry {}>'.format(self.header)

    def kmercounts(self, k):
        if k < 1 or k > 10:
            raise ValueError('k must be between 1 and 10 inclusive')
        return _kmercounts(self.sequence, k)

    def fourmer_freq(self):
        return _fourmerfreq(self.sequence)

def byte_iterfasta(filehandle, comment=b'#'):
    linemask = bytes.maketrans(b'acgtuUswkmyrbdhvnSWKMYRBDHV',
                               b'ACGTTTNNNNNNNNNNNNNNNNNNNNN')

    # Make it work for persistent iterators, e.g. lists
    line_iterator = iter(filehandle)
    # Skip to first header
    try:
        for probeline in line_iterator:
            stripped = probeline.lstrip()
            if stripped.startswith(comment):
                pass

            elif probeline[0:1] == b'>':
                break

            else:
                raise ValueError('First non-comment line is not a Fasta header')

        else: # no break
            raise ValueError('Empty or outcommented file')

    except TypeError:
        errormsg = 'First line does not contain bytes. Are you reading file in binary mode?'
        raise TypeError(errormsg) from None

    header = probeline[1:-1].decode()
    buffer = list()

    # Iterate over lines
    for line in line_iterator:
        if line.startswith(comment):
            pass

        elif line.startswith(b'>'):
            yield FastaEntry(header, bytearray().join(buffer))
            buffer.clear()
            header = line[1:-1].decode()

        else:
            buffer.append(line)

    yield FastaEntry(header, bytearray().join(buffer))

def c_kmercounts(bytesarray, k: int, counts):
    """Count tetranucleotides of contig and put them in counts vector.
    The bytearray is expected to be np.uint8 of bytevalues of the contig.
    The counts is expected to be an array of 4^k 32-bit integers with value 0.
    """

    kmer = 0
    character = 0
    charvalue = 0
    i = 0
    countdown = k-1
    contiglength = len(bytesarray)
    mask = (1 << (2 * k)) - 1
    lut = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 0, 4, 1, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                              4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    for i in range(contiglength):
        character = bytesarray[i]
        charvalue = lut[character]

        if charvalue == 4:
            countdown = k

        kmer = ((kmer << 2) | charvalue) & mask

        if countdown == 0:
            counts[kmer] += 1
        else:
            countdown -= 1

def _kmercounts(sequence: bytearray, k: int):
    """Returns a 32-bit integer array containing the count of all kmers
    in the given bytearray.
    Only Kmers containing A, C, G, T (bytes 65, 67, 71, 84) are counted"""

    if k > 10 or k < 1:
        return ValueError('k must be between 1 and 10, inclusive.')

    counts = _np.zeros(4**k)

    sequenceview = sequence
    countview = counts

    c_kmercounts(sequenceview, k, countview)

    return counts

def c_fourmer_freq(counts, result):
    """Puts kmercounts of k=4 in a nonredundant vector.

    The result is expected to be a 136 32-bit float vector
    The counts is expected to be an array of 256 32-bit integers
    """

    countsum = 0
    i = 0
    # Lookup in this array gives the index of the canonical tetranucleotide.
    # E.g CCTA is the 92nd alphabetic 4mer, whose reverse complement, TAGG, is the 202nd.
    # So the 92th and 202th value in this array is the same.
    # Hence we can map 256 4mers to 136 normal OR reverse-complemented ones
    complementer_fourmer = [0, 1, 2, 3, 4, 5, 6, 7,
            8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 11, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 23, 42, 43, 44, 7, 45, 46,
            47, 48, 49, 50, 51, 34, 52, 53, 54, 19, 55, 56, 57, 3, 58, 59,
            60, 57, 61, 62, 63, 44, 64, 65, 66, 30, 67, 68, 69, 14, 70, 71,
            72, 54, 73, 74, 75, 41, 76, 77, 78, 26, 79, 80, 66, 10, 81, 82,
            83, 51, 84, 85, 86, 37, 87, 88, 75, 22, 89, 90, 63, 6, 91, 92,
            93, 47, 94, 95, 83, 33, 96, 97, 72, 18, 98, 99, 60, 2, 100,
            101, 99, 56, 102, 103, 90, 43, 104, 105, 80, 29, 106, 107, 68,
            13, 108, 109, 97, 53, 110, 111, 88, 40, 112, 113, 77, 25, 114,
            105, 65, 9, 115, 116, 95, 50, 117, 118, 85, 36, 119, 111, 74,
            21, 120, 103, 62, 5, 121, 122, 92, 46, 123, 116, 82, 32, 124,
            109, 71, 17, 125, 101, 59, 1, 126, 125, 98, 55, 127, 120, 89,
            42, 128, 114, 79, 28, 129, 106, 67, 12, 130, 124, 96, 52, 131,
            119, 87, 39, 132, 112, 76, 24, 128, 104, 64, 8, 133, 123, 94,
            49, 134, 117, 84, 35, 131, 110, 73, 20, 127, 102, 61, 4, 135,
            121, 91, 45, 133, 115, 81, 31, 130, 108, 70, 16, 126, 100, 58, 0]

    for i in range(256):
        countsum += counts[i]

    if countsum == 0:
        return

    floatsum = countsum

    for i in range(256):
        result[complementer_fourmer[i]] += counts[i] / floatsum


def _fourmerfreq(sequence: bytearray):
    """Returns float32 array of 136-length float32 representing the
    tetranucleotide (fourmer) frequencies of the DNA.
    Only fourmers containing A, C, G, T (bytes 65, 67, 71, 84) are counted"""

    counts = _np.zeros(256)
    frequencies = _np.zeros(136)

    sequenceview = sequence
    fourmercountview = counts
    frequencyview = frequencies

    c_kmercounts(sequenceview, 4, fourmercountview)
    c_fourmer_freq(fourmercountview, frequencyview)

    return frequencies













#-------------------------------------------------------------------------------------------------------------------------#
#                                BAM FILES HERUNDER                                                                       #
#-------------------------------------------------------------------------------------------------------------------------#







DEFAULT_SUBPROCESSES = min(8, _os.cpu_count())

def read_bamfiles(paths, dumpdirectory=None, refhash=None, minscore=None, minlength=None,
                  minid=None, subprocesses=DEFAULT_SUBPROCESSES, logfile=None):
    "Placeholder docstring - replaced after this func definition"

    # Define callback function depending on whether a logfile exists or not
    if logfile is not None:
        def _callback(result):
            path, rpkms, length = result
            print('\tProcessed', path, file=logfile)
            logfile.flush()

    else:
        def _callback(result):
            pass

    # Bam files must be unique.
    if len(paths) != len(set(paths)):
        raise ValueError('All paths to BAM files must be unique.')

    # Bam files must exist
    for path in paths:
        if not _os.path.isfile(path):
            raise FileNotFoundError(path)

    if dumpdirectory is not None:
        # Dumpdirectory cannot exist, but its parent must exist
        dumpdirectory = _os.path.abspath(dumpdirectory)
        if _os.path.exists(dumpdirectory):
            raise FileExistsError(dumpdirectory)

        parentdir = _os.path.dirname(_os.path.abspath(dumpdirectory))
        if not _os.path.isdir(parentdir):
            raise FileNotFoundError("Parent dir of " + dumpdirectory)

        # Create directory to dump in
        _os.mkdir(dumpdirectory)

    # Spawn independent processes to calculate RPKM for each of the BAM files
    processresults = list()

    # Queue all the processes
    with _multiprocessing.Pool(processes=subprocesses) as pool:
        for pathnumber, path in enumerate(paths):
            if dumpdirectory is None:
                outpath = None
            else:
                outpath = _os.path.join(dumpdirectory, str(pathnumber) + '.npz')

            arguments = (path, outpath, refhash, minscore, minlength, minid)
            processresults.append(pool.apply_async(_get_contig_rpkms, arguments,
                                                   callback=_callback))

        all_done, any_fail = False, False
        while not (all_done or any_fail):
            _time.sleep(5)
            all_done = all(process.ready() and process.successful() for process in processresults)
            any_fail = any(process.ready() and not process.successful() for process in processresults)

            if all_done:
                pool.close() # exit gently
            if any_fail:
                pool.terminate() # exit less gently

        # Wait for all processes to be cleaned up
        pool.join()

    # Raise the error if one of them failed.
    for path, process in zip(paths, processresults):
        if process.ready() and not process.successful():
            print('\tERROR WHEN PROCESSING:', path, file=logfile)
            print('Vamb aborted due to error in subprocess. See stacktrace for source of exception.')
            if logfile is not None:
                logfile.flush()
            process.get()
            print('process.get()')

    ncontigs = None
    for processresult in processresults:
        path, rpkm, length = processresult.get()

        # Verify length of contigs are same for all BAM files
        if ncontigs is None:
            ncontigs = length
        elif length != ncontigs:
            raise ValueError('First BAM file has {} headers, {} has {}.'.format(
                             ncontigs, path, length))

    # If we did not dump to disk, load directly from process results to
    # one big matrix...
    if dumpdirectory is None:
        columnof = {p:i for i, p in enumerate(paths)}
        rpkms = _np.zeros((ncontigs, len(paths)), dtype=_np.float32)

        for processresult in processresults:
            path, rpkm, length = processresult.get()
            rpkms[:, columnof[path]] = rpkm

    # If we did, instead merge them from the disk
    else:
        dumppaths = [_os.path.join(dumpdirectory, str(i) + '.npz') for i in range(len(paths))]
        rpkms = mergecolumns(dumppaths)

    return rpkms

def _get_contig_rpkms(inpath, outpath, refhash, minscore, minlength, minid):
    """Returns  RPKM (reads per kilobase per million mapped reads)
    for all contigs present in BAM header.

    Inputs:
        inpath: Path to BAM file
        outpath: Path to dump depths array to or None
        refhash: Expected reference hash (None = no check)
        minscore: Minimum alignment score (AS field) to consider
        minlength: Discard any references shorter than N bases
        minid: Discard any reads with ID lower than this

    Outputs:
        path: Same as input path
        rpkms:
            If outpath is not None: None
            Else: A float32-array with RPKM for each contig in BAM header
        length: Length of rpkms array
        hash: md5 of reference names
    """

    bamfile = _pysam.AlignmentFile(inpath, "rb")
    _check_bamfile(inpath, bamfile, refhash, minlength)
    counts = count_reads(bamfile, minscore, minid)
    rpkms = calc_rpkm(counts, bamfile.lengths, minlength)
    bamfile.close()

    # If dump to disk, array returned is None instead of rpkm array
    if outpath is not None:
        arrayresult = None
        _np.savez_compressed(outpath, rpkms)
    else:
        arrayresult = rpkms

    return inpath, arrayresult, len(rpkms)

def mergecolumns(pathlist):
    """Merges multiple npz files with columns to a matrix.

    All paths must be npz arrays with the array saved as name 'arr_0',
    and with the same length.

    Input: pathlist: List of paths to find .npz files to merge
    Output: Matrix with one column per npz file
    """

    if len(pathlist) == 0:
        return _np.array([], dtype=_np.float32)

    for path in pathlist:
        if not _os.path.exists(path):
            raise FileNotFoundError(path)

    first = _np.load(pathlist[0])['arr_0']
    length = len(first)
    ncolumns = len(pathlist)

    result = _np.zeros((length, ncolumns), dtype=_np.float32)
    result[:,0] = first

    for columnno, path in enumerate(pathlist[1:]):
        column = _np.load(path)['arr_0']
        if len(column) != length:
            raise ValueError("Length of data at {} is not equal to that of "
                             "{}".format(path, pathlist[0]))
        result[:,columnno + 1] = column

    return result

def _check_bamfile(path, bamfile, refhash, minlength):
    "Checks bam file for correctness (refhash and sort order). To be used before parsing."
    # If refhash is set, check ref hash matches what is found.
    if refhash is not None:
        if minlength is None:
            refnames = bamfile.references
        else:
            pairs = zip(bamfile.references, bamfile.lengths)
            refnames = (ref for (ref, len) in pairs if len >= minlength)

        hash = _hash_refnames(refnames)
        if hash != refhash:
            errormsg = ('BAM file {} has reference hash {}, expected {}. '
                        'Verify that all BAM headers and FASTA headers are '
                        'identical and in the same order.')
            raise ValueError(errormsg.format(path, hash.hex(), refhash.hex()))

    # Check that file is unsorted or sorted by read name.
    hd_header = bamfile.header.get("HD", dict())
    sort_order = bamfile.header.get("SO") #hd_header.get("SO")
    if sort_order in ("coordinate", "unknown"):
        errormsg = ("BAM file {} is marked with sort order '{}', must be "
                    "unsorted or sorted by readname.")
        raise ValueError(errormsg.format(path, sort_order))

def count_reads(bamfile, minscore=None, minid=None):
    """Count number of reads mapping to each reference in a bamfile,
    optionally filtering for score and minimum id.
    Multi-mapping reads MUST be consecutive in file, and their counts are
    split among the references.

    Inputs:
        bamfile: Open pysam.AlignmentFile
        minscore: Minimum alignment score (AS field) to consider [None]
        minid: Discard any reads with ID lower than this [None]

    Output: Float32 Numpy array of read counts for each reference in file.
    """
    # Use 64-bit floats for better precision when counting
    readcounts = _np.zeros(len(bamfile.lengths))

    # Initialize with first aligned read - return immediately if the file
    # is empty
    filtered_segments = _filter_segments(bamfile, minscore, minid)
    try:
        segment = next(filtered_segments)
        read_name = segment.query_name
        multimap = 1.0
        reference_ids = [segment.reference_id]
    except StopIteration:
        return readcounts.astype(_np.float32)

    # Now count up each read in the BAM file
    for segment in filtered_segments:
        # If we reach a new read_name, we tally up the previous read
        # towards all its references, split evenly.
        if segment.query_name != read_name:
            read_name = segment.query_name
            to_add = 1.0 / multimap
            for reference_id in reference_ids:
                readcounts[reference_id] += to_add
            reference_ids.clear()
            multimap = 0.0

        multimap += 1.0
        reference_ids.append(segment.reference_id)

    # Add final read
    to_add = 1.0 / multimap
    for reference_id in reference_ids:
        readcounts[reference_id] += to_add

    return readcounts.astype(_np.float32)

def calc_rpkm(counts, lengths, minlength=None):
    """Calculate RPKM based on read counts and sequence lengths.

    Inputs:
        counts: Numpy vector of read counts from count_reads
        lengths: Iterable of contig lengths in same order as counts
        minlength [None]: Discard any references shorter than N bases

    Output: Float32 Numpy vector of RPKM for all seqs with length >= minlength
    """
    lengtharray = _np.array(lengths)
    if len(counts) != len(lengtharray):
        raise ValueError("counts length and lengths length must be same")

    millionmappedreads = counts.sum() / 1e6

    # Prevent division by zero
    if millionmappedreads == 0:
        rpkm = _np.zeros(len(lengtharray), dtype=_np.float32)
    else:
        kilobases = lengtharray / 1000
        rpkm = (counts / (kilobases * millionmappedreads)).astype(_np.float32)

    # Now filter away small contigs
    if minlength is not None:
        lengthmask = lengtharray >= minlength
        rpkm = rpkm[lengthmask]

    return rpkm

def _hash_refnames(refnames):
    "Hashes an iterable of strings of reference names using MD5."
    hasher = _md5()
    for refname in refnames:
        hasher.update(refname.encode().rstrip())

    return hasher.digest()

def _filter_segments(segmentiterator, minscore, minid):
    """Returns an iterator of AlignedSegment filtered for reads with low
    alignment score.
    """

    for alignedsegment in segmentiterator:
        # Skip if unaligned or suppl. aligment
        if alignedsegment.flag & 0x804 != 0:
            continue

        if minscore is not None and alignedsegment.get_tag('AS') < minscore:
            continue

        if minid is not None and _identity(alignedsegment) < minid:
            continue

        yield alignedsegment

def _identity(segment):
    "Return the nucleotide identity of the given aligned segment."
    mismatches, matches = 0, 0
    for kind, number in segment.cigartuples:
        # 0, 7, 8, is match/mismatch, match, mismatch, respectively
        if kind in (0, 7, 8):
            matches += number
        # 1, 2 is insersion, deletion
        elif kind in (1, 2):
            mismatches += number
    matches -= segment.get_tag('NM')
    return matches / (matches+mismatches)