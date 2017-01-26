from helper import FileHelper, Word2VecHelper
from optparse import OptionParser
import helper

helper.enableLog()

def _run(args):
    print(args)
    if (args.force == 1):
        FileHelper.generateDataFile()
    if (args.embedding == 1):
        files = ["./train/{}/{}/positive.txt".format(args.ontology, args.type),
                 "./train/{}/{}/negative.txt".format(args.ontology, args.type)]
        Word2VecHelper.createModel(files=files, name="{}_{}".format(args.ontology, args.type))

if __name__ == '__main__':

    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="dbpedia")
    parser.add_option('-t', '--type', dest='type', default="generic")
    parser.add_option('-f', '--force', dest='force', default=0, type=int)
    parser.add_option('-w', '--embedding', dest='embedding', type=int, default=1)
    opts, args = parser.parse_args()

    _run(opts)
