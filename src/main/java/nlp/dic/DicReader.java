package nlp.dic;

import org.nlpcn.commons.lang.util.logging.Log;
import org.nlpcn.commons.lang.util.logging.LogFactory;

import java.io.*;

/**
 * 加载词典用的类
 * 
 * @author ansj
 */
public class DicReader {

	private static final Log logger = LogFactory.getLog();

	public static BufferedReader getReader(String name) throws FileNotFoundException {
		// maven工程修改词典加载方式
		InputStream in = new FileInputStream(new File("./src/main/java/resources/"+name));
		try {
			return new BufferedReader(new InputStreamReader(in, "UTF-8"));
		} catch (UnsupportedEncodingException e) {
			logger.warn("不支持的编码", e);
		}
		return null;
	}

	public static InputStream getInputStream(String name) {
		// maven工程修改词典加载方式
		InputStream in = DicReader.class.getResourceAsStream("/" + name);
		return in;
	}
}
