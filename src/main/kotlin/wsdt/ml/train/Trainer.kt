package wsdt.ml.train

import org.apache.ant.compress.taskdefs.Unzip
import org.datavec.api.io.filters.BalancedPathFilter
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.FileSplit
import org.datavec.api.split.InputSplit
import org.datavec.image.loader.BaseImageLoader
import org.datavec.image.loader.NativeImageLoader
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.Updater
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.util.ModelSerializer
import org.deeplearning4j.zoo.PretrainedType
import org.deeplearning4j.zoo.model.VGG16
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.io.IOException
import java.util.*

object Trainer {

    private val seed: Long = 12345
    val RAND_NUM_GEN = Random(seed)
    val ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS
    var LABEL_GENERATOR_MAKER = ParentPathLabelGenerator()
    var PATH_FILTER = BalancedPathFilter(RAND_NUM_GEN, ALLOWED_FORMATS, LABEL_GENERATOR_MAKER)

    private const val EPOCH = 5 //5
    private const val BATCH_SIZE = 16
    private const val TRAIN_SIZE = 85
    private const val NUM_POSSIBLE_LABELS = 2

    private const val SAVING_INTERVAL = 100 //100

    var DATA_PATH = "resources"
    val TRAIN_FOLDER = "$DATA_PATH/train_both"
    val TEST_FOLDER = "$DATA_PATH/test_both"
    private val SAVING_PATH = "$DATA_PATH/saved/modelIteration_"

    private val FREEZE_UNTIL_LAYER = "fc2"

    //private static final String DATA_URL = "https://github.com/mhw32/derm-ai/raw/master/trained_models/model_best.pth.tar"; //https://dl.dropboxusercontent.com/s/tqnp49apphpzb40/dataTraining.zip?dl=0";

    @Throws(IOException::class)
    fun unzip(fileZip: File) {

        val unzipper = Unzip()
        unzipper.setSrc(fileZip)
        unzipper.setDest(File(DATA_PATH))
        unzipper.execute()
    }

    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val zooModel = VGG16()
        print("Start Downloading VGG16 model...")
        val preTrainedNet = zooModel.initPretrained(PretrainedType.IMAGENET) as ComputationGraph
        print(preTrainedNet.summary())

        /*print("Start Downloading Data...");

        downloadAndUnzipDataForTheFirstTime();*/
        print("Data unzipped")
        // Define the File Paths
        val trainData = File(TRAIN_FOLDER)
        val testData = File(TEST_FOLDER)
        val train = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN)
        val test = FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN)


        val sample = train.sample(PATH_FILTER, TRAIN_SIZE.toDouble(), (100 - TRAIN_SIZE).toDouble())
        val trainIterator = getDataSetIterator(sample[0])
        val devIterator = getDataSetIterator(sample[1])


        val fineTuneConf = FineTuneConfiguration.Builder()
                .learningRate(5e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(seed)
                .build()

        val vgg16Transfer = TransferLearning.GraphBuilder(preTrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(FREEZE_UNTIL_LAYER)
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(NUM_POSSIBLE_LABELS)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), FREEZE_UNTIL_LAYER)
                .build()
        vgg16Transfer.setListeners(ScoreIterationListener(5))
        print(vgg16Transfer.summary())

        val testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1.0, 0.0)[0])
        var iEpoch = 0
        var i = 0
        while (iEpoch < EPOCH) {
            while (trainIterator.hasNext()) {
                val trained = trainIterator.next()
                vgg16Transfer.fit(trained)
                if (i % SAVING_INTERVAL == 0 && i != 0) {

                    ModelSerializer.writeModel(vgg16Transfer, File(SAVING_PATH + i + "_epoch_" + iEpoch + ".zip"), false)
                    evalOn(vgg16Transfer, devIterator, i)
                }
                i++
            }
            trainIterator.reset()
            iEpoch++

            evalOn(vgg16Transfer, testIterator, iEpoch)
        }
    }


    @Throws(IOException::class)
    fun evalOn(vgg16Transfer: ComputationGraph, testIterator: DataSetIterator, iEpoch: Int) {
        print("Evaluate model at iteration $iEpoch ....")
        val eval = vgg16Transfer.evaluate(testIterator)
        print(eval.stats())
        testIterator.reset()

    }

    @Throws(IOException::class)
    fun getDataSetIterator(sample: InputSplit): DataSetIterator {

        val imageRecordReader = ImageRecordReader(224, 224, 3, LABEL_GENERATOR_MAKER)
        imageRecordReader.initialize(sample)

        val iterator = RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, NUM_POSSIBLE_LABELS)
        iterator.preProcessor = VGG16ImagePreProcessor()
        return iterator
    }


}
