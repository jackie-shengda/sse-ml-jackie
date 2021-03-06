package sdtest;

import nlp.tokenizations.tokenizerFactory.ChineseTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;

import java.io.*;

/**@author wangfeng
 * @date June 3,2017
 * @Description
 *
 */

public class ChineseTokenizerTest {

    private final String toTokenize = "文章，1984年6月26日出生于陕西省西安市，中国内地男演员、导演。2006年毕业于中央戏剧学院表演系。\n" +
            "2004年参演电视剧《与青春有关的日子》，开始在影视圈崭露头角[1]  。2005年拍摄古装剧《锦衣卫》。2007年主演赵宝刚导演的青春剧《奋斗》；[2]  同年，主演首部电影《走着瞧》。2008年主演滕华涛执导的电视剧《蜗居》，饰演80后城市青年小贝。[1]  2009年，在电影《海洋天堂》中扮演自闭症患者王大福；同年参演抗战题材的电视剧《雪豹》[4]  。2011年，主演的电视剧《裸婚时代》在各大卫视播出；[5]  2011年-2012年连续2年获得北京大学生电影节[6-7]  最受大学生欢迎男演员奖。2012年，凭借电影《失恋33天》获得第31届大众电影百花奖最佳男主角奖；[8]  同年成立自己经营的北京君竹影视文化有限公司，并导演第一部影视作品《小爸爸》。2013年2月，主演的电影《西游·降魔篇》在全国上映。[9] \n" +
            "2014年3月28日，主演的中韩合资文艺爱情片《我在路上最爱你》在全国上映。2014年12月18日，在姜文执导的动作喜剧片《一步之遥》中扮演武七一角。[10]  2016年，主演电视剧《少帅》，饰演张学良[11]  ；主演电视剧《剃刀边缘》[12]  。7月15日导演的电影《陆垚知马俐》上映。[13] \n" +
            "演艺事业外，文章也参与公益慈善事业，2010年成立大福自闭症关爱基金。";
    private final String[] expect = {"青山绿水", "和", "伟大", "的", "科学家", "让", "世界", "更", "美好", "和平"};

    //正面语料处理
    @Test
    public void testChineseTokenizer() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(new File("./src/main/java/resources/正面.txt")));
        File target = new File("./src/main/java/resources/positive_nospace.txt");
        if(!target.exists()){
            target.createNewFile();
        }
        FileWriter fw = new FileWriter(target);
        TokenizerFactory tokenizerFactory = new ChineseTokenizerFactory();
        String lineStr = br.readLine();
        int j=1;
        while (lineStr!=null){
            fw.write("正面 ");
            System.out.println("line "+ j + " " +lineStr);
            Tokenizer tokenizer = tokenizerFactory.create(lineStr);
            for (int i = 0; i < tokenizer.countTokens(); ++i) {
                String nextToken = tokenizer.nextToken();
                System.out.println(nextToken);
                fw.write(nextToken+" ");
//            assertEquals(nextToken, expect[i]);
            }
            fw.write("\r\n");
            lineStr = br.readLine();
            j++;
        }
        fw.flush();
        fw.close();
    }

    //负面语料处理
    @Test
    public void negativeChineseTokenizer() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(new File("./src/main/java/resources/负面.txt")));
        File target = new File("./src/main/java/resources/negative_nospace.txt");
        if(!target.exists()){
            target.createNewFile();
        }
        FileWriter fw = new FileWriter(target);
        TokenizerFactory tokenizerFactory = new ChineseTokenizerFactory();
        String lineStr = br.readLine();;
        while (lineStr!=null){
            fw.write("负面 ");
            Tokenizer tokenizer = tokenizerFactory.create(lineStr);
            for (int i = 0; i < tokenizer.countTokens(); ++i) {
                String nextToken = tokenizer.nextToken();
//                System.out.println(nextToken);
                fw.write(nextToken+" ");
//            assertEquals(nextToken, expect[i]);
            }
            fw.write("\r\n");
            lineStr = br.readLine();
        }
        fw.flush();
        fw.close();
    }

    @Test
    public void findMaxlength() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(new File("./src/main/java/resources/corpus2.txt")));
        String lineStr = br.readLine();
        int len=0;
        while (lineStr!=null){
            if(lineStr.length()>len) len=lineStr.length();
            lineStr = br.readLine();
        }
        System.out.print(len);
    }

    @Test
    public void dealWithCorpus() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(new File("./src/main/java/resources/正面.txt")));
        BufferedReader br2 = new BufferedReader(new FileReader(new File("./src/main/java/resources/负面.txt")));
        File target = new File("./src/main/java/resources/corpus5.txt");
        if(!target.exists()){
            target.createNewFile();
        }
        FileWriter fw = new FileWriter(target);
        TokenizerFactory tokenizerFactory = new ChineseTokenizerFactory();
        String lineStr = br.readLine();
        String lineStr2 = br2.readLine();
        while (lineStr!=null){
            if(lineStr.length()>10 && lineStr.length()<50){
                fw.write("正面 ");
                fw.write(lineStr);
                fw.write("\r\n");
            }
            lineStr = br.readLine();
        }
        while (lineStr2!=null){
            if(lineStr2.length()>10 && lineStr2.length()<50){
                fw.write("负面 ");
                fw.write(lineStr2);
                fw.write("\r\n");
            }
            lineStr2 = br2.readLine();
        }
        fw.flush();
        fw.close();
    }

    @Test
    public void dealWithIssue() throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(new File("./src/main/java/resources/issue/issues.txt")));
        String lineStr = br.readLine();
        int l = lineStr.length();
        while (lineStr!=null){
            if(lineStr.length()>l){
                l=lineStr.length();
            }
            lineStr = br.readLine();
        }
        System.out.println(l);
    }
}
