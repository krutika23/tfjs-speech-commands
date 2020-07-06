import { Component } from '@angular/core';
import * as tf from '@tensorflow/tfjs'
import * as SpeechComm from './src';
@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'tfjs-angular';
  num_frames:number = 5; //change number
  examples = [];
  input_shape = [5, 232, 1];   //change number
  model:any;
  recognizer:any = SpeechComm.create('BROWSER_FFT');
  	  
  

	  listen(){
		  this.title=this.title+' inside listen';
		   this.recognizer.ensureModelLoaded();
		   this.recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
		   this.title='speak';
		   const vals = this.normalize(data.subarray(-frameSize * this.num_frames));
		   const input = tf.tensor(vals, [1, ...this.input_shape]);
		   // console.log('input:', input);
		   const model1 =await tf.loadLayersModel('http://localhost:4200/assets/my-model.json');
		   //this.title=model1;
		   const probs = model1.predict(input);
		   const predLabel = (probs as tf.Tensor).argMax(1);
		   this.moveSlider(predLabel);
		   tf.dispose([input, probs, predLabel]);
		 }, {
		   overlapFactor: 0.83,
		   includeSpectrogram: true,
		   probabilityThreshold: 0.85
		 });
		   //this.title=this.title+' end listen ';
	  }


	moveSlider(labelTensor) {
		console.log(labelTensor.toString()[12]);
		 const label:number = ( labelTensor.data())[0];
		 if(labelTensor.toString()[12]==="0"){
		 	console.log('aqua ');
		 	this.title=this.title+' Aqua';
		 }
		 else if (labelTensor.toString()[12]==="1"){
		 	console.log('Start');
		 	this.title=this.title+' Start';
		 }
		 else if (labelTensor.toString()[12]==="2"){
		 	console.log('search');
		 	this.title=this.title+' Search';
		 }
		 else if (labelTensor.toString()[12]==="3"){
		 	console.log('reset');
		 	this.title=this.title+' Reset';
		 }
		

	     
	}
	 normalize(x) {
		 const mean = -100;
		 const std = 10;
		 return x.map(x => (x - mean) / std);
	}
}
