package ml;

import ml.text.functions.TextPipeline;
import nlp.tokenizations.tokenizerFactory.ChineseTokenizerFactory;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Jackie on 2017/8/13.
 */
public class commentModelTest {
    private static int VOCAB_SIZE = 15000;
    private static int maxCorpusLength = 64;  //最大语料长度
    private static int numLabel = 2;           //标签个数
    private static int batchSize  = 50;        //批处理大小
    private static int totalEpoch = 3;        //样本训练次数

    public static void main(String args[]) throws Exception {
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]")
                .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                .set("spark.kryo.registrator","org.nd4j.Nd4jRegistrator")
                .set("spark.executor.memory", "10g")
                .set("spark.driver.memory","10g")
                .setAppName("NLP Java Spark");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);

        File modelPath = new File("./src/main/java/resources/commentModel.zip");

        //加载模型
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelPath);
        model.getLabels();

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
        JavaRDD<String> javaRDDCorpus = jsc.textFile("./src/main/java/resources/corpus6.txt");
        TextPipeline textPipeLineCorpus = new TextPipeline(javaRDDCorpus, broadcasTokenizerVarMap);
        JavaRDD<List<String>> javaRDDCorpusToken = textPipeLineCorpus.tokenize();   //tokenize the corpus
        textPipeLineCorpus.buildVocabCache();                                       //build and get the vocabulary
        textPipeLineCorpus.buildVocabWordListRDD();                                 //build corpus
        Broadcast<VocabCache<VocabWord>> vocabCorpus = textPipeLineCorpus.getBroadCastVocabCache();
        JavaRDD<List<VocabWord>> javaRDDVocabCorpus = textPipeLineCorpus.getVocabWordListRDD(); //get tokenized corpus

//语料标签
        JavaRDD<String> javaRDDLabel = jsc.textFile("./src/main/java/resources/label.txt");
        TextPipeline textPipelineLabel = new TextPipeline(javaRDDLabel, broadcasTokenizerVarMap);   //broadcasTokenizerVarMap 需要视情况重新定义
        JavaRDD<List<String>> javaRDDCorpusLabel = textPipeLineCorpus.tokenize();
        textPipelineLabel.buildVocabCache();
        textPipelineLabel.buildVocabWordListRDD();
        Broadcast<VocabCache<VocabWord>> vocabLabel = textPipelineLabel.getBroadCastVocabCache();
        JavaRDD<List<VocabWord>> javaRDDVocabLabel = textPipelineLabel.getVocabWordListRDD();

        //转换为训练集
        JavaRDD<Tuple2<List<VocabWord>,VocabWord>> javaPairRDDVocabLabel = javaRDDCorpusToken.map(new Function<List<String>, String>() {
            @Override
            public String call(List<String> strings) throws Exception {
                if(strings.get(0).equals("是")){
                    System.out.println(strings);
                }
                return strings.get(0);
            }
        }).zip(javaRDDVocabCorpus).map(new Function<Tuple2<String, List<VocabWord>>, Tuple2<List<VocabWord>,VocabWord>>() {
                                           @Override
                                           public Tuple2<List<VocabWord>,VocabWord> call(Tuple2<String, List<VocabWord>> stringListTuple2) throws Exception {
                                               String token = stringListTuple2._1();
                                               VocabCache<VocabWord> vocabLabel1 = vocabLabel.getValue();
                                               VocabWord word = vocabLabel1.wordFor(token);
                                               if(!token.equals("正面") && !token.equals("负面")){
                                                   System.out.println("===========================================   ERROR：  "+token);
                                               }
                                               return new Tuple2(stringListTuple2._2().subList(1,stringListTuple2._2.size()),word);
                                           }

                                       }
        );

        JavaRDD<DataSet> javaRDDTrainData = javaPairRDDVocabLabel.map(new Function<Tuple2<List<VocabWord>,VocabWord>, DataSet>() {

            @Override
            public DataSet call(Tuple2<List<VocabWord>, VocabWord> tuple) throws Exception {
                List<VocabWord> listWords = tuple._1;
                VocabWord labelWord = tuple._2;
                INDArray features = Nd4j.create(1, 1, maxCorpusLength);
                INDArray labels = Nd4j.create(1, (int)numLabel, maxCorpusLength);
                INDArray featuresMask = Nd4j.zeros(1, maxCorpusLength);
                INDArray labelsMask = Nd4j.zeros(1, maxCorpusLength);
                int[] origin = new int[3];
                int[] mask = new int[2];
                origin[0] = 0;                        //arr(0) store the index of batch sentence
                mask[0] = 0;
                int j = 0;
//                System.out.print("**********  listWords size: " + listWords.size() + "    "+ listWords.get(0).getWord()+listWords.get(1).getWord());
//                System.out.println("**********  labelWord:      " + labelWord.getWord());
                for (VocabWord vw : listWords) {         //traverse the list which store an entire sentence
                    origin[2] = j;
//                    System.out.print(vw.getWord()+" ");
//                    System.out.print("origin: "+origin[0]+" "+origin[1]+" "+origin[2]);
                    features.putScalar(origin, vw.getIndex());
                    //
                    mask[1] = j;
//                    System.out.println("    mask: "+mask[0]+" "+mask[1]+"   word: "+vw.getWord());
                    featuresMask.putScalar(mask, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
                    ++j;
                }
//                System.out.println();
                //
                int lastIdx = listWords.size();
                int idx = labelWord.getIndex();
                labels.putScalar(new int[]{0,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
                labelsMask.putScalar(new int[]{0,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
                return new DataSet(features, labels, featuresMask, labelsMask);
            }
        });

        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
                .workerPrefetchNumBatches(0)
                .saveUpdater(false)
                .averagingFrequency(5)
                .batchSizePerWorker(batchSize)
                .build();
        SparkDl4jMultiLayer sparknet = new SparkDl4jMultiLayer(jsc, model.getLayerWiseConfigurations(), trainMaster);

        List<String> labelList  = new ArrayList<>();
        labelList.add("负面");
        labelList.add("正面");
        int total = 100;
        int count = 0;
//        for(int i=0;i<total;i++){
            Evaluation eval  = sparknet.evaluate(javaRDDTrainData,labelList);
            System.out.println(eval.stats());
//            if(eval.stats().contains("Examples labeled as 正面 classified by model as 负面")){
//                count++;
//            }
//        }
//        System.out.println("预测成功率： "+ new Double(count/total));
    }

}
