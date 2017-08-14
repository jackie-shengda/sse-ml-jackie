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
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import scala.Tuple2;

import java.io.File;
import java.io.Serializable;
import java.util.*;

/**
 * Created by Jackie.S on 2017/7/25.
 */
public class Trainer implements Serializable{

    private int VOCAB_SIZE = 16000;
    private int maxCorpusLength = 64;  //最大语料长度
    private int numLabel = 2;           //标签个数
    private int batchSize  = 45;        //批处理大小
    private int totalEpoch = 20;        //样本训练次数

    public void tainning() throws Exception {
        System.out.println(Runtime.getRuntime().maxMemory());
//        System.setProperty("hadoop.home.dir", "G:\\environment\\hadoop");

//        Configuration hadoopConf = new Configuration();
//        hadoopConf.set("fs.hdfs.impl",org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[*]")
                .set("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                .set("spark.kryo.registrator","org.nd4j.Nd4jRegistrator")
                .set("spark.Kryoserializer.buffer.max","4096m")
                .set("spark.executor.memory", "10g")
                .set("spark.driver.memory","10g")
                .setAppName("NLP Java Spark");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        jsc.hadoopConfiguration().set("fs.hdfs.impl",org.apache.hadoop.hdfs.DistributedFileSystem.class.getName());

        MultiLayerConfiguration netconf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .iterations(1)
                .learningRate(0.01)
                .learningRateScoreBasedDecayRate(0.5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true)
                .l2(5 * 1e-4)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(VOCAB_SIZE).nOut(256).activation("identity").build())
                .layer(1, new GravesLSTM.Builder().nIn(256).nOut(256).activation("softsign").build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax").nIn(256).nOut(2).build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.recurrent(VOCAB_SIZE))
                .build();

        ArrayList<String> stopWords = new ArrayList<>();
        stopWords.add(",");
        stopWords.add("是");
        stopWords.add("的");
        stopWords.add("我");
        stopWords.add("就");
//        stopWords.add("【");
//        stopWords.add("】");
        Map<String, Object> TokenizerVarMap = new HashMap<>();      //定义文本处理的各种属性
        TokenizerVarMap.put("numWords", 1);     //词最小出现次数
        TokenizerVarMap.put("nGrams", 1);       //language model parameter
        TokenizerVarMap.put("tokenizer", ChineseTokenizerFactory.class.getName());  //分词器实现
        TokenizerVarMap.put("tokenPreprocessor", CommonPreprocessor.class.getName());
        TokenizerVarMap.put("useUnk", true);    //unlisted words will use usrUnk
        TokenizerVarMap.put("vectorsConfiguration", new VectorsConfiguration());
        TokenizerVarMap.put("stopWords", stopWords);  //stop words
        Broadcast<Map<String, Object>>  broadcasTokenizerVarMap = jsc.broadcast(TokenizerVarMap);   //broadcast the parameter map

        //训练语料分词
        JavaRDD<String> javaRDDCorpus = jsc.textFile("./src/main/java/resources/corpus5.txt");      //情感分类
//        JavaRDD<String> javaRDDCorpus = jsc.textFile("./src/main/java/resources/issue/issues.txt");      //issue分类
        TextPipeline textPipeLineCorpus = new TextPipeline(javaRDDCorpus, broadcasTokenizerVarMap);
        JavaRDD<List<String>> javaRDDCorpusToken = textPipeLineCorpus.tokenize();   //tokenize the corpus
        textPipeLineCorpus.buildVocabCache();                                       //build and get the vocabulary
        textPipeLineCorpus.buildVocabWordListRDD();                                 //build corpus
        Broadcast<VocabCache<VocabWord>> vocabCorpus = textPipeLineCorpus.getBroadCastVocabCache();
        JavaRDD<List<VocabWord>> javaRDDVocabCorpus = textPipeLineCorpus.getVocabWordListRDD(); //get tokenized corpus

        //语料标签
        JavaRDD<String> javaRDDLabel = jsc.textFile("./src/main/java/resources/label.txt");         //情感分类
//        JavaRDD<String> javaRDDLabel = jsc.textFile("./src/main/java/resources/issue/labels.txt");         //issue分类
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
//                                               if(!token.equals("正面") && !token.equals("负面")){
//                                                   System.out.println("===========================================   ERROR：  "+token);
//                                               }
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
//                System.out.println("idx: "+idx+"    lastIdx: "+lastIdx);
//                System.out.println(idx+": "+labelWord.getWord());
                labels.putScalar(new int[]{0,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
                labelsMask.putScalar(new int[]{0,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
                return new DataSet(features, labels, featuresMask, labelsMask);
            }
        });

        ParameterAveragingTrainingMaster trainMaster = new ParameterAveragingTrainingMaster.Builder(batchSize)
                .workerPrefetchNumBatches(0)
                .saveUpdater(true)
                .averagingFrequency(5)
                .batchSizePerWorker(batchSize)
                .build();
        SparkDl4jMultiLayer sparknet = new SparkDl4jMultiLayer(jsc, netconf, trainMaster);
        sparknet.setListeners(Collections.<IterationListener>singletonList(new ScoreIterationListener(1)));
/*

        //初始化用户界面后端
        UIServer uiServer = UIServer.getInstance();

        //设置网络信息（随时间变化的梯度、分值等）的存储位置。这里将其存储于内存。
        StatsStorage statsStorage = new InMemoryStatsStorage();         //或者： new FileStatsStorage(File)，用于后续的保存和载入

        //将StatsStorage实例连接至用户界面，让StatsStorage的内容能够被可视化
        uiServer.attach(statsStorage);

        //然后添加StatsListener来在网络定型时收集这些信息
        sparknet.setListeners(new StatsListener(statsStorage));
*/

        for( int numEpoch = 0; numEpoch < totalEpoch; ++numEpoch){
            sparknet.fit(javaRDDTrainData);
            Evaluation evaluation = sparknet.evaluate(javaRDDTrainData);
            double accuracy = evaluation.accuracy();
            System.out.println("====================================================================");
            System.out.println("Epoch " + numEpoch + " Has Finished");
            System.out.println("Accuracy: " + accuracy);
            System.out.println("====================================================================");
            MultiLayerNetwork network = sparknet.getNetwork();
            File out = new File("./src/main/java/resources/trainingModel.zip");
            ModelSerializer.writeModel(network, out, true);
        }
//
//        MultiLayerNetwork network = sparknet.getNetwork();
       /* FileSystem hdfs = FileSystem.get(jsc.hadoopConfiguration());
        Path hdfsPath = new Path("./src/main/java/resources/trainingModel.zip");
        if( hdfs.exists(hdfsPath) ){
            hdfs.delete(hdfsPath, true);
        }*/
//        File out = new File("./src/main/java/resources/trainingModel.zip");
//        FSDataOutputStream outputStream = hdfs.create(hdfsPath);
//        ModelSerializer.writeModel(network, out, true);

         /*---Finish Saving the Model------*/
//        String VocabCorpusPath = "./src/main/java/resources/courpus.dat";               //语料保存地址
//        String VocabLabelPath = "./src/main/java/resources/label.dat";                   //标签保存地址
//        VocabCache<VocabWord> saveVocabCorpus = vocabCorpus.getValue();
//        VocabCache<VocabWord> saveVocabLabel = vocabLabel.getValue();
//        SparkUtils.writeObjectToFile(VocabCorpusPath, saveVocabCorpus, jsc);
//        SparkUtils.writeObjectToFile(VocabLabelPath, saveVocabLabel, jsc);

    }

}
