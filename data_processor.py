import vamb_tools


def get_tnfs(path=None):
    if path is None:
        path = 'E:/Repositories/MI108F20_Binning/test/Bin.gz'
    with vamb_tools.Reader(path, 'rb') as filehandle:
        tnfs = vamb_tools.read_contigs(filehandle, minlength=4)
    return tnfs


def get_depth(paths=None):
    if paths is None:
        paths = ['E:/Repositories/MI108F20_Binning/test/one.bam', 'E:/Repositories/MI108F20_Binning/test/two.bam',
                 'E:/Repositories/MI108F20_Binning/test/three.bam']
    return vamb_tools.read_bamfiles(paths)
