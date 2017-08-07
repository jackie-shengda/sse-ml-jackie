package nlp.recognition.impl;

import java.util.List;

import nlp.domain.Result;
import nlp.domain.Term;
import nlp.library.SynonymsLibrary;
import nlp.recognition.Recognition;
import org.nlpcn.commons.lang.tire.domain.SmartForest;

/**
 * 同义词功能
 * 
 * @author Ansj
 *
 */
public class SynonymsRecgnition implements Recognition {

	private static final long serialVersionUID = 5961499108093950130L;

	private SmartForest<List<String>> synonyms = null;

	public SynonymsRecgnition() {
		this.synonyms = SynonymsLibrary.get();
	}

	public SynonymsRecgnition(String key) {
		this.synonyms = SynonymsLibrary.get(key);
	}
	
	public SynonymsRecgnition(SmartForest<List<String>> synonyms){
		this.synonyms = synonyms ;
	}

	@Override
	public void recognition(Result result) {
		for (Term term : result) {
			SmartForest<List<String>> branch = synonyms.getBranch(term.getName());
			if (branch != null && branch.getStatus() > 1) {
				List<String> syns = branch.getParam();
				if (syns != null) {
					term.setSynonyms(syns);
				}
			}
		}
	}

}
