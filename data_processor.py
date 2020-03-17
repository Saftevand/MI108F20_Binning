import vamb_tools


def get_tnfs(path=None):
    if path is None:
        path = 'test/bigfasta.fna.gz'
    with vamb_tools.Reader(path, 'rb') as filehandle:
        tnfs = vamb_tools.read_contigs(filehandle, minlength=4)
    return tnfs


def get_depth(paths=None):
    if paths is None:
        paths = ['test/one.bam', 'test/two.bam',
                 'test/three.bam']
    return vamb_tools.read_bamfiles(paths)
