package org.fisheries.ml;

import java.io.File;
import java.util.Random;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.MultiImageTransform;
import org.datavec.image.transform.ScaleImageTransform;
import org.datavec.image.transform.WarpImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

/**
 * This class generates the training and test dataset iterator for an image dataset.
 * The input image files should follow the standard directory structure. This
 * class also applies transforms on the images to standardize the image size and
 * to increase the number of data points.
 *
 */
public class ImageDataSetIteratorGenerator {

	private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;

	private static final long seed = 12345;

	private static final Random randNumGen = new Random(seed);

	private final int height;
	private final int width;
	private final int channels;
	private final int outputNum;

	private InputSplit trainData;
	private InputSplit testData;
	private ParentPathLabelGenerator labelMaker;
	private DataNormalization scaler;

	public ImageDataSetIteratorGenerator(String dataRootDir, int height, int width, int channels, int outputNum) {
		this.height = height;
		this.width = width;
		this.channels = channels;
		this.outputNum = outputNum;

		File parentDir = new File(dataRootDir);
		if(!parentDir.exists()) {
			
		}
		FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, randNumGen);

		//You do not have to manually specify labels. This class (instantiated as below) will
		//parse the parent dir and use the name of the subdirectories as label/class names
		labelMaker = new ParentPathLabelGenerator();
		//The balanced path filter gives you fine tune control of the min/max cases to load for each class
		//Below is a bare bones version. Refer to javadocs for details
		BalancedPathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedExtensions, labelMaker);

		//Split the image files into train and test. Specify the train test split as 80%,20%
		InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 80, 20);
		trainData = filesInDirSplit[0];
		testData = filesInDirSplit[1];
		scaler = new ImagePreProcessingScaler(0,1);
	}

	public DataSetIterator getTrainingDataSetIterator(int batchSize) throws Exception {
		ImageRecordReader trainRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
		ImageTransform transform = new MultiImageTransform(randNumGen,
				new FlipImageTransform(0), new FlipImageTransform(1),
				new ScaleImageTransform(10), new WarpImageTransform(10));

		//Initialize the record reader with the train data and the transform chain
		trainRecordReader.initialize(trainData,transform);
		//convert the record reader to an iterator for training
		DataSetIterator trainDataIter = new RecordReaderDataSetIterator(trainRecordReader, batchSize, 1, outputNum);
        scaler.fit(trainDataIter);
        trainDataIter.setPreProcessor(scaler);
		return trainDataIter;
	}

	public DataSetIterator getTestDataSetIterator(int batchSize) throws Exception {
		ImageRecordReader testRecordReader = new ImageRecordReader(height,width,channels,labelMaker);
		ImageTransform transform = new MultiImageTransform(randNumGen,
				new FlipImageTransform(0), new FlipImageTransform(1),
				new ScaleImageTransform(10), new WarpImageTransform(10));

		//Initialize the record reader with the test data and the transform chain
		testRecordReader.initialize(testData,transform);
		//convert the record reader to an iterator for training
		DataSetIterator testDataIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 1, outputNum);
        scaler.fit(testDataIter);
        testDataIter.setPreProcessor(scaler);
		return testDataIter;
	}
	
}
