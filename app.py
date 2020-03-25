from flask import Flask, render_template,request
import pickle
import numpy as np

app=Flask(__name__)

word_list=pickle.load(open('mystrings.pkl','rb'))

clf=pickle.load(open('model.pkl','rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email=request.form.get('email')

    input_mail = []
    for i in word_list:
        input_mail.append(email.count(i[0]))

    x=clf.predict(np.array(input_mail).reshape(1, 3000))

    x=x[0]

    return render_template('index.html', label=str(x))


if __name__=="__main__":
    app.run(debug=True)