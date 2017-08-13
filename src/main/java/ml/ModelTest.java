package ml;

import ml.text.functions.TextPipeline;
import nlp.tokenizations.tokenizerFactory.ChineseTokenizerFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Jackie on 2017/8/13.
 */
public class ModelTest  {
    public void test() throws Exception {
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]")
                .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                .set("spark.kryo.registrator","org.nd4j.Nd4jRegistrator")
                .set("spark.executor.memory", "10g")
                .set("spark.driver.memory","10g")
                .setAppName("NLP Java Spark");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        File modelPath = new File("./src/main/java/resources/trainingModel.zip");

        //加载模型
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        model.getLabels();

        Evaluation eval = new Evaluation(2);

        ArrayList<String> stopWords = new ArrayList<>();
        stopWords.add(",");
        stopWords.add("是");
        stopWords.add("的");
        stopWords.add("我");
        stopWords.add("就");
        Map<String, Object> TokenizerVarMap = new HashMap<>();      //定义文本处理的各种属性
        TokenizerVarMap.put("numWords", 1);     //词最小出现次数
        TokenizerVarMap.put("nGrams", 1);       //language model parameter
        TokenizerVarMap.put("tokenizer", ChineseTokenizerFactory.class.getName());  //分词器实现
        TokenizerVarMap.put("tokenPreprocessor", CommonPreprocessor.class.getName());
        TokenizerVarMap.put("useUnk", true);    //unlisted words will use usrUnk
        TokenizerVarMap.put("vectorsConfiguration", new VectorsConfiguration());
        TokenizerVarMap.put("stopWords", stopWords);  //stop words
        Broadcast<Map<String, Object>>  broadcasTokenizerVarMap = jsc.broadcast(TokenizerVarMap);   //broadcast the parameter map


        //获取训练数据
        JavaRDD<String> javaRDDCorpus = jsc.textFile("./src/main/java/resources/corpus4_min.txt");
        TextPipeline textPipeLineCorpus = new TextPipeline(javaRDDCorpus, broadcasTokenizerVarMap);
        JavaRDD<List<String>> javaRDDCorpusToken = textPipeLineCorpus.tokenize();   //tokenize the corpus
        textPipeLineCorpus.buildVocabCache();                                       //build and get the vocabulary
        textPipeLineCorpus.buildVocabWordListRDD();                                 //build corpus
        Broadcast<VocabCache<VocabWord>> vocabCorpus = textPipeLineCorpus.getBroadCastVocabCache();
        JavaRDD<List<VocabWord>> javaRDDVocabCorpus = textPipeLineCorpus.getVocabWordListRDD(); //get tokenized corpus


    }

}
