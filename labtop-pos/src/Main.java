import java.io.*;
import edu.stanford.nlp.util.logging.Redwood;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.List;

import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class Main {
    public static void main (String [] args) {
        String doc = txt2String(new File("./labtop-data-to-pos.txt"));
        doc = doc.trim();
        String [] lines = doc.split(System.getProperty("line.separator"));
        System.out.println("line: " + lines.length);
        System.out.println(lines[0]);

        StringBuilder posedDoc = new StringBuilder();

        MaxentTagger tagger = new MaxentTagger("models/english-left3words-distsim.tagger");
        for (String sentence : lines) {
            String posedSentence = tagger.tagString(sentence);
            posedSentence = posedSentence.trim();
            posedDoc.append(posedSentence);
            posedDoc.append(System.getProperty("line.separator"));
        }
//        List<List<HasWord>> sentences = null;
//        try {
//            sentences = MaxentTagger.tokenizeText(new BufferedReader(new FileReader("./labtop-data-to-pos.txt")));
//            System.out.println("len:" + sentences.size());
//            S
//        writeToFile(posedDoc.toString().trim(), new File("./labtop-data-posed.txt"));
//        System.out.println("end");ystem.out.println(sentences.get(0));
//            for (List<HasWord> sentence : sentences) {
//                List<TaggedWord> tSentence = tagger.tagSentence(sentence);
//                String posedSentence = SentenceUtils.listToString(tSentence, false);
//                posedDoc.append(posedSentence);
//                posedDoc.append(System.getProperty("line.separator"));
////                System.out.println(SentenceUtils.listToString(sentence, false));
////                System.out.println(SentenceUtils.listToString(tSentence, false));
////                System.out.println("####################");
//            }
//        } catch (FileNotFoundException e) {
//            e.printStackTrace();
//        }


    }

    public static String txt2String(File file){
        StringBuilder result = new StringBuilder();
        try{
            BufferedReader br = new BufferedReader(new FileReader(file));//构造一个BufferedReader类来读取文件
            String s = null;
            while((s = br.readLine())!=null){//使用readLine方法，一次读一行
                result.append(System.lineSeparator()+s);
            }
            br.close();
        }catch(Exception e){
            e.printStackTrace();
        }
        return result.toString();
    }

    public static void writeToFile (String doc, File file) {
        FileWriter fw = null;
        try {
            fw = new FileWriter(file);
            fw.write(doc);
            fw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fw != null) {
                try {
                    fw.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

    }

}

