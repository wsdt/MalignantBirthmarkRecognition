package wsdt.ml.predict

import org.datavec.api.split.FileSplit
import org.datavec.image.loader.NativeImageLoader
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor
import wsdt.ml.train.Trainer

import java.io.File
import java.io.FileInputStream
import java.io.IOException

/** Heart of the neural network. Performs training and prediction operations.
 * Therefore, it is used by the Predictor and Trainer.
 *
 * This project uses DeepLearning4J and TransferLearning. Thus, it uses the VGG-16
 * model from Oxford University. */
class VG16 {

    /** Classifies the picture uploaded from the user. */
    @Throws(IOException::class)
    fun detectType(file: File): INDArray {
        if (computationGraph == null) {
            computationGraph = loadModel()
        }

        computationGraph!!.init()
        println(computationGraph!!.summary())
        val loader = NativeImageLoader(224, 224, 3)
        val image = loader.asMatrix(FileInputStream(file))
        val scaler = VGG16ImagePreProcessor()
        scaler.transform(image)
        return computationGraph!!.outputSingle(false, image)
    }

    /** After training it is evaluated via an own test set.
     * Test set is located in /resources/test_both */
    @Throws(IOException::class)
    private fun runOnTestSet() {
        val computationGraph = loadModel()
        val trainData = File(Trainer.TEST_FOLDER)
        val test = FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, Trainer.RAND_NUM_GEN)
        val inputSplit = test.sample(Trainer.PATH_FILTER, 100.0, 0.0)[0]
        val dataSetIterator = Trainer.getDataSetIterator(inputSplit)
        Trainer.evalOn(computationGraph, dataSetIterator, 1)
    }

    /** Loads model from file path system.
     * @return ComputationGraph: Representation of already trained model*/
    @Throws(IOException::class)
    fun loadModel(): ComputationGraph {
        computationGraph = ModelSerializer.restoreComputationGraph(File(TRAINED_PATH_MODEL))
        return computationGraph as ComputationGraph
    }

    /** Will run after training cycle to evaluate quality. */
    companion object {
        private val TRAINED_PATH_MODEL = Trainer.DATA_PATH + "/saved/modelIteration_100_epoch_0.zip"
        private var computationGraph: ComputationGraph? = null

        @Throws(IOException::class)
        @JvmStatic
        fun main(args: Array<String>) {
            VG16().runOnTestSet()
        }
    }

}
