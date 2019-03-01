package wsdt.ml.predict

open class BirthmarkType(confidence: Double) {
    val _confidence:Double = confidence
}

class BasalCellCarcinoma(confidence: Double) : BirthmarkType(confidence) {
    companion object {
        val NAME = "Basal Cell Carcinoma"
        val STR_ID = "BASAL_CELL_CARCINOMA"
    }
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}

class Melanoma(confidence: Double) : BirthmarkType(confidence) {
    companion object {
        val NAME = "Melanoma"
        val STR_ID = "MELANOMA"
    }
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}

class SquamousCellCarcinoma(confidence: Double) : BirthmarkType(confidence) {
    companion object {
        val NAME = "Squamous Cell Carcinoma"
        val STR_ID = "SQUAMOUS_CELL_CARCINOMA"
    }
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}

class NotKnown(confidence: Double) : BirthmarkType(confidence) {
    companion object {
        val NAME = "Not Known"
        val STR_ID = "NOT_KNOWN"
    }
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}