from helper import FileHelper, Word2VecHelper
from optparse import OptionParser

if __name__ == '__main__':

    parser = OptionParser('''%prog -o ontology -t type -f force ''')
    parser.add_option('-o', '--ontology', dest='ontology', default="yago")
    parser.add_option('-t', '--type', dest='type', default="specific")
    parser.add_option('-f', '--force', dest='force', default=1, type=int)
    parser.add_option('-c', '--classifier', dest='classifier', default='nb')
    parser.add_option('-j', '--job', dest='job', type=int, default=10)
    parser.add_option('-w', '--window', dest='window', type=int, default=2)
    parser.add_option('-s', '--size', dest='size', type=int, default=300)
    parser.add_option('-m', '--min', dest='min_count', type=int, default=5)
    parser.add_option('-w', '--embedding', dest='experiment', type=int, default=0)
    opts, args = parser.parse_args()

    if(opts.f == 1) :
        FileHelper.generateDataFile()
    if (opts.embedding == 1):
        files = ["./train/{}/{}/positive.txt".format(args.ontology, args.type),
                 "./train/{}/{}/negative.txt".format(args.ontology, args.type)]
        Word2VecHelper.createModel(files=files, name="{}_{}".format(args.ontology,args.type))
