package wsdt.ml.predict

/** Base type for all birthmark classifications.
 * @param confidence: The confidence represents the certainty of how sure the neural network is that the uploaded image
 * belongs to that specific category. All confidence values of all sub-classes should equal to 1 (= 100%). */
open class BirthmarkType(confidence: Double) {
    val _confidence:Double = confidence
}

/** Class Benign to categorize harmless birthmarks.
 * @return BirthmarkType(confidence): Returns sub-class instance nested in parent class BirthmarkType. */
class Benign(confidence: Double) : BirthmarkType(confidence) {
    /** Defines several meta information such as NAME for labelling the confidence in the UI as well
     * as STR_ID to perform further categorization operations. */
    companion object {
        val NAME = "Benign"
        val STR_ID = "BENIGN"
    }
    /** Returns json, which will be nested into a json-array and is going to be returned after every user request.
     * @return String: Json-Object as String to nest into json-array. */
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}

/** Class Malignant to categorize harmful birthmarks.
 * @return BirthmarkType(confidence): Returns sub-class instance nested in parent class BirthmarkType. */
class Malignant(confidence: Double) : BirthmarkType(confidence) {
    /** Defines several meta information such as NAME for labelling the confidence in the UI as well
     * as STR_ID to perform further categorization operations. */
    companion object {
        val NAME = "Malignant"
        val STR_ID = "MALIGNANT"
    }
    /** Returns json, which will be nested into a json-array and is going to be returned after every user request.
     * @return String: Json-Object as String to nest into json-array. */
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}

/** Class NotKnown to categorize birthmarks which cannot be correctly assigned
 * (e.g. uploading pictures of other objects than birthmarks).
 * @return BirthmarkType(confidence): Returns sub-class instance nested in parent class BirthmarkType. */
class NotKnown(confidence: Double) : BirthmarkType(confidence) {
    /** Defines several meta information such as NAME for labelling the confidence in the UI as well
     * as STR_ID to perform further categorization operations. */
    companion object {
        val NAME = "Not Known"
        val STR_ID = "NOT_KNOWN"
    }
    /** Returns json, which will be nested into a json-array and is going to be returned after every user request.
     * @return String: Json-Object as String to nest into json-array. */
    override fun toString(): String {
        return "{\"id\":\"$STR_ID\", \"name\":\"$NAME\", \"confidence\":\"${this._confidence}\"}"
    }
}