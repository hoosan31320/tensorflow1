import React from 'react'
import { Text, View } from 'react-native'
import * as tf from '@tensorflow/tfjs'
import { bundleResourceIO } from '@tensorflow/tfjs-react-native'

class App extends React.Component {
   state = {
     isTfReady: false,
     model: false,
   }

   async componentDidMount() {
     await tf.ready()
     this.setState({ isTfReady: true })

     const modelJSON = require('./assets/model/model.json');
     const modelWeights = require('./assets/model/weights.bin');
    //  const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights));
    //  model.summary();

    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    model.summary();

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);
    
    // Train the model using the data.
    model.fit(xs, ys).then(() => {
      // Use the model to do inference on a data point the model hasn't seen before:
      model.predict(tf.tensor2d([5], [1, 1])).print();
    });


     this.setState({ model })
   }

   render() {
     return (
       <View style={{
         flex: 1,
         justifyContent: 'center',
         alignItems: 'center'
       }}>
         <Text>
           TF: {this.state.isTfReady ? "Ready" : "Waiting"}
         </Text>
         <Text>
           MODEL: {this.state.model ? "Ready" : "Waiting"}
         </Text>
       </View>
     )
   }
}

export default App
