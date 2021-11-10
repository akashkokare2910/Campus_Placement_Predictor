from flask import Flask, request, render_template
import pickle


# unplickling the model

file = open('campusplacementpredictor.pkl', 'rb')
rf = pickle.load(file)
file.close()


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def hello_world():

    if request.method == 'POST':

        mydict = request.form
        gender = int(mydict['gender'])
        spec = int(mydict['spec'])
        tech = int(mydict['tech'])  
        work = int(mydict['work'])
        ssc = float(mydict['ssc'])
        hsc = float(mydict['hsc'])
        dsc = float(mydict['dsc'])
        mba = float(mydict['mba'])
        inputfeatures = [[gender, spec, tech, work, ssc, hsc, dsc, mba]]

        # predicting the class either 0 or 1

        predictedclass = rf.predict(inputfeatures)

        # predicting the probability

        predictedprob = rf.predict_proba(inputfeatures)

        print(predictedclass, predictedprob[0][0])

        if predictedclass[0] == 1:
            proba = predictedprob[0][1]

        else:
            proba = predictedprob[0][0]

        print(predictedclass, proba*100)

        placemap = {1: 'Will be Placed', 0: 'Better Luck Next Time :('}
        predictedclasssend = placemap[predictedclass[0]]

        if predictedclass[0] == 1:
            return render_template('show.html', predictedclasssend=predictedclasssend, predictedprob=round(proba*100, 2), placed=True)

        else:
            return render_template('show.html', predictedclasssend=predictedclasssend)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
