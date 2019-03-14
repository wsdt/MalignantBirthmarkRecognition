package wsdt

import wsdt.ml.predict.Predictor

/** Run this file to classify a test image called 'test.jpg'. */
object Run {
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        Predictor.predict("test")
    }
}
