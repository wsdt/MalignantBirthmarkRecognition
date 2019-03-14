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

/** Used to train VGG-16 model via transfer learning.
 * Will output fine-tuned models in /resources/saved/ */
object Trainer {
    /** Seed for random num generator. */
    private val seed: Long = 12345
    /** Used to generate random numbers, e.g. to set random
     * weights for non-frozen layers. */
    val RAND_NUM_GEN = Random(seed)
    /** Accesses image file formats, which are allowed to train the
     * neural network. */
    val ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS
    var LABEL_GENERATOR_MAKER = ParentPathLabelGenerator()
    var PATH_FILTER = BalancedPathFilter(RAND_NUM_GEN, ALLOWED_FORMATS, LABEL_GENERATOR_MAKER)

    /** Configure training algorithm. */
    private const val EPOCH = 5 //5
    private const val BATCH_SIZE = 16
    private const val TRAIN_SIZE = 85
    /** Number of labels, in our case: Benign and Malignant */
    private const val NUM_POSSIBLE_LABELS = 2

    /** In which intervals should the model saved (in case
     * of exceptions [e.g. OutOfMemory, ...]) */
    private const val SAVING_INTERVAL = 100 //100

    /** Where are general resource files located? */
    var DATA_PATH = "resources"
    /** Location of trainable image data set (already extracted, etc.) */
    val TRAIN_FOLDER = "$DATA_PATH/train_both"
    /** Location of test image sets, to evaluate how precise the neural
     * network is after the training cycle. */
    val TEST_FOLDER = "$DATA_PATH/test_both"
    /** Where and in which naming-pattern to save the trained models. */
    private val SAVING_PATH = "$DATA_PATH/saved/modelIteration_"

    /** Freeze VGG-16 layer until which layer? */
    private val FREEZE_UNTIL_LAYER = "fc2"

    /** Main function to train a model with predefined data sets.
     * After the training cycle the neural network will be tested against
     * a test imageset. */
    @Throws(IOException::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val zooModel = VGG16()
        print("Start Downloading VGG16 model...")
        val preTrainedNet = zooModel.initPretrained(PretrainedType.IMAGENET) as ComputationGraph
        print(preTrainedNet.summary())

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


    /** Executes on every iteration and gives corresponding output.
     * @param vgg16Transfer: Current model to test.
     * @param testIterator: Current iteration respectively iterator.
     * @param iEpoch: Current epoch. */
    @Throws(IOException::class)
    fun evalOn(vgg16Transfer: ComputationGraph, testIterator: DataSetIterator, iEpoch: Int) {
        print("Evaluate model at iteration $iEpoch ....")
        val eval = vgg16Transfer.evaluate(testIterator)
        print(eval.stats())
        testIterator.reset()

    }

    /** Reads image according to desired channels and dimensions.
     * @param sample: Current inputSplit to initialize ImageRecordReader. */
    @Throws(IOException::class)
    fun getDataSetIterator(sample: InputSplit): DataSetIterator {
        val imageRecordReader = ImageRecordReader(224, 224, 3, LABEL_GENERATOR_MAKER)
        imageRecordReader.initialize(sample)

        val iterator = RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, NUM_POSSIBLE_LABELS)
        iterator.preProcessor = VGG16ImagePreProcessor()
        return iterator
    }


}
