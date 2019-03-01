package wsdt

import wsdt.ml.predict.Predictor

object Run {

    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        Predictor.predict("test")
    }
}
