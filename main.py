import streamlit as st
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding
from music21 import converter
from pretty_midi import PrettyMIDI

index_to_char = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: '#', 5: '%', 6: '&', 7: "'", 8: '(', 9: ')', 10: '+', 11: ',', 12: '-', 13: '.', 14: '/', 15: '0', 16: '1', 17: '2', 18: '3', 19: '4', 20: '5', 21: '6', 22: '7', 23: '8', 24: '9', 25: ':', 26: '=', 27: '?', 28: 'A', 29: 'B', 30: 'C', 31: 'D', 32: 'E', 33: 'F', 34: 'G', 35: 'H', 36: 'I', 37: 'J', 38: 'K', 39: 'L', 40: 'M', 41: 'N', 42: 'O', 43: 'P', 44: 'Q', 45: 'R', 46: 'S', 47: 'T', 48: 'U', 49: 'V', 50: 'W', 51: 'X', 52: 'Y', 53: '[', 54: '\\', 55: ']', 56: '^', 57: '_', 58: 'a', 59: 'b', 60: 'c', 61: 'd', 62: 'e', 63: 'f', 64: 'g', 65: 'h', 66: 'i', 67: 'j', 68: 'k', 69: 'l', 70: 'm', 71: 'n', 72: 'o', 73: 'p', 74: 'q', 75: 'r', 76: 's', 77: 't', 78: 'u', 79: 'v', 80: 'w', 81: 'x', 82: 'y', 83: 'z', 84: '|', 85: '~'}

def make_model(unique_chars):
    model = Sequential()
    
    model.add(Embedding(input_dim = unique_chars, output_dim = 512, batch_input_shape = (1, 1))) 
  
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(256, return_sequences = True, stateful = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(256, stateful = True)) 
    #remember, that here we haven't given return_sequences = True because here we will give only one character to generate the
    #sequence. In the end, we just have to get one output which is equivalent to getting output at the last time-stamp. So, here
    #in last layer there is no need of giving return sequences = True.
    model.add(Dropout(0.2))
    
    model.add((Dense(unique_chars)))
    model.add(Activation("softmax"))
    
    return model

def generate_sequence(epoch_num, initial_index, seq_length):
    unique_chars = len(index_to_char)
    
    model = make_model(unique_chars)
    model.load_weights("model/" + "weights.{}.h5".format(epoch_num*10))
     
    sequence_index = [initial_index]
    
    for _ in range(seq_length):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]
        
        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(unique_chars), size = 1, p = predicted_probs)
        
        sequence_index.append(sample[0])
    
    seq = ''.join(index_to_char[c] for c in sequence_index)
    
    cnt = 0
    for i in seq:
        cnt += 1
        if i == "\n":
            break
    seq1 = seq[cnt:]
    #above code is for ignoring the starting string of a generated sequence. This is because we are passing any arbitrary 
    #character to the model for generating music. Now, the model start generating sequence from that character itself which we 
    #have passed, so first few characters before "\n" contains meaningless word. Model start generating the music rhythm from
    #next line onwards. The correct sequence it start generating from next line onwards which we are considering.
    
    cnt = 0
    for i in seq1:
        cnt += 1
        if i == "\n" and seq1[cnt] == "\n":
            break
    seq2 = seq1[:cnt]
    #Now our data contains three newline characters after every tune. So, the model has leart that too. So, above code is used for
    #ignoring all the characters that model has generated after three new line characters. So, here we are considering only one
    #tune of music at a time and finally we are returning it.
    
    return seq2





# ------------------------------------Streamlit App------------------------------------

displayGenre = ('Orchestra', 'Grand Piano', 'Acoustic Piano', 'Percussion' ,'Keyboard')
optionsGenre = list(range(len(displayGenre)))

st.title("AI Music Generator")

modelNo = st.slider("Select the model you want to use", 1, 10, 10)
st.caption("Smaller number will generate more Errors in music.")
st.text("")

initialChar = st.number_input("Enter initial character", 0, 85, step=1, value=5)
st.caption("This will be given as initial character to model for generating sequence.")
st.text("")

length = st.slider("Length of music sequence", 100, 1000, 400)
st.caption("Too small number will generate hardly generate any musical sequence.")
st.text("")

genre = st.selectbox("Genre", optionsGenre, index=2,format_func=lambda x: displayGenre[x])

sampleRate = st.selectbox("Sampling Rate", [88200, 66150, 44100, 22050, 11025], index=1) 

st.text("")

if(st.button('Generate Music')):
    attempt=0
    music=None
    musicMIDI=None
    with st.sidebar:
        with st.spinner("Work in progress..."):
            while musicMIDI==None and attempt<=6:
                try:
                    attempt+=1

                    # Generate ABC Notation
                    music = generate_sequence(modelNo, initialChar, length)
                    abcFile = open('output.abc', 'w')
                    abcFile.write(music)
                    abcFile.close()

                    # Convert output.abc ---> output.mid
                    s = converter.parse('output.abc')
                    s.quarterLength = 0.5
                    s.write('midi', fp='output.mid')
            

                    # Convert output.mid ---> output.wav
                    musicMIDI = PrettyMIDI(midi_file="output.mid")
                    fileName = "SoundFont0" + str(genre+1) + ".sf2"
                    wav = musicMIDI.fluidsynth(sf2_path=fileName)

                    st.text_area("", value=music, height=400)
                    st.audio(wav, sample_rate=sampleRate)
                    break
                except:
                    pass
            
            if attempt>6:
                abcFile = open("output.abc", "r")
                abcFile_contents = abcFile.read()
                abcFile.close()        


                # Convert output.abc ---> output.mid
                s = converter.parse('defaultOutput.abc')
                s.quarterLength = 0.5
                s.write('midi', fp='output.mid')
            

                # Convert output.mid ---> output.wav
                musicMIDI = PrettyMIDI(midi_file="output.mid")
                fileName = "SoundFont0" + str(genre+1) + ".sf2"
                wav = musicMIDI.fluidsynth(sf2_path=fileName)

                st.text_area("", value=abcFile_contents, height=400)
                st.audio(wav, sample_rate=sampleRate)

                
                

# --------------------------------------------------------------------------------------------
