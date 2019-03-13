package wsdt.ml.predict

open class BirthmarkType(confidence: Double) {
    val _confidence:Double = confidence
}

class Benign(confidence: Double) : BirthmarkType(confidence) {
    companion object {
        val NAME = "Benign"
        val STR_ID = "BENIGN"
    }
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}

class Malignant(confidence: Double) : BirthmarkType(confidence) {
    companion object {
        val NAME = "Malignant"
        val STR_ID = "MALIGNANT"
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