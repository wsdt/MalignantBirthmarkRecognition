
import wsdt.ml.predict.BirthmarkType
import wsdt.ml.predict.Predictor
import java.io.File
import java.util.*
import javax.servlet.annotation.WebServlet
import javax.servlet.http.HttpServlet
import javax.servlet.http.HttpServletRequest
import javax.servlet.http.HttpServletResponse

@WebServlet(name="Analyze", value=["/analyze"])
class PictureController : HttpServlet() {
    private val charPool : List<Char> = ('a'..'z') + ('A'..'Z') + ('0'..'9')

    override fun doPost(req: HttpServletRequest, res: HttpServletResponse) {
        val fileName = decodeImg(req.getParameter("image"))
        val birthmarkTypes:List<BirthmarkType> = Predictor.predict(fileName, this.servletContext.getRealPath("/WEB-INF/classes"))

        val resultJson = "[${birthmarkTypes[0]},${birthmarkTypes[1]},${birthmarkTypes[2]}]"
        res.writer.write(resultJson)
    }

    fun getRandomStr() : String {
        return (1..30)
                .map { kotlin.random.Random.nextInt(0, charPool.size) }
                .map(charPool::get)
                .joinToString("")

    }

    fun decodeImg(base64Str: String) : String {
        val fileName = getRandomStr()

        val imageByteArr =  Base64.getDecoder().decode(base64Str)

        val pathname = this.servletContext.getRealPath("/WEB-INF/classes/uploaded/$fileName.jpg")
        File(pathname).writeBytes(imageByteArr)

        return fileName
    }
}
