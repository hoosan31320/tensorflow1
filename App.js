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
     const model = await tf.loadLayersModel(bundleResourceIO(modelJSON, modelWeights));
     model.summary();
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
