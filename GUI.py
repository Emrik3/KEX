from tkinter import *
from dataProcessing import translations_to_word_classes
from GrammarTests import predict




def process_predict(file):
    #translations_to_word_classes(file, 'WC_input.json')
    return predict('TM_all_2nd', file, 'WC_input.json', 2)

def get_input():
    """Takes in text and and runs the appropriate program to generate the output"""
    value = input_text.get("1.0", "end-1c")
    with open("user_input.txt", "w") as outfile:
        outfile.write(value)
    result = process_predict("user_input.txt")
    output_text.insert("end", result)

def options():
    """No functionality yet just shows what will eventually become buttons for the different analysis which
        could be done"""
    frame = Frame(win)
    frame.pack(side=LEFT, fill = Y)
    Label(frame, text="Order of Markov Chain").pack()
    Label(frame, text="1").pack(anchor=W)
    Label(frame, text="2").pack(anchor=W)
    Label(frame, text="3").pack(anchor=W)
    Label(frame, text="4").pack(anchor=W)
    Label(frame, text="5").pack(anchor=W)
    Label(frame, text="6").pack(anchor=W) 

    frame2 = Frame(win)
    frame2.pack(side=LEFT, fill = Y)
    Label(frame2, text="Norm/classic method").pack()
    Label(frame2, text="1-norm").pack(anchor=W)
    Label(frame2, text="2-norm").pack(anchor=W)
    Label(frame2, text="inf-norm").pack(anchor=W)
    Label(frame2, text="Frobenius-norm").pack(anchor=W)
    Label(frame2, text="Maximum likely hood").pack(anchor=W)
    Label(frame2, text="Cross Entropy").pack(anchor=W)

    fram3 = Frame(win)
    fram3.pack(side=LEFT, fill = Y)
    Label(fram3, text="Machine learning").pack()
    Label(fram3, text="LSTM?").pack(anchor=W)
    Label(fram3, text="Hidden markov chain?").pack(anchor=W)
    Label(fram3, text="etc?").pack(anchor=W)
    Label(fram3, text="etc?").pack(anchor=W)

    fram4 = Frame(win)
    fram4.pack(side=LEFT, fill = Y)
    Label(fram4, text="Create academic vocabulary").pack()
    Label(fram4, text="Find grammatical mistakes").pack(anchor=W)
    Label(fram4, text="Get a grammar score").pack(anchor=W)
    Label(fram4, text="Generate transition matrix" ).pack(anchor=W)
    Label(fram4, text="etc?").pack(anchor=W)


win=Tk()
win.geometry("1000x400")
options()
input_text = Text(win, height = 5,width = 40, bd = 2, relief="groove")
input_text.pack()
button = Button(win, height = 1, width = 30,text = "Execute", command = get_input)
button.pack()

output_text = Text(win, height = 5, width = 40, bd = 2, relief="groove")
output_text.pack()


win.mainloop()
