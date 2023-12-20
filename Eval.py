import Levenshtein as Lev
import matplotlib.pyplot as plt

def main():
    stop = 10087
    i = 1

    accY = []
    accNoCapY = []
    avgAccY = []
    avgAccNoCapY = []
    avgDetY = []
    avgDetNoCapY = []

    total = 0
    totalDetect = 0
    avgAc = 0
    avgAcNoCap = 0

    captionPath = './dataset/gt.txt'
    gtFile = open(captionPath, "r", encoding="utf8")

    while i <= stop:
        gt = gtFile.readline().split(sep=",")
        gt = [l.strip() for l in gt]

        # only considers Latin script
        if gt[1] == "Latin":
            total += 1
            print(gt)

            outfilePath = './output/word_' + str(i) + '.txt'
            f = open(outfilePath, "r")
            text = f.readlines()
            text = [t.strip() for t in text]
            f.close()

            line = " ".join(text)
            print(line)
            word = "" if len(line) == 0 else line
            target = gt[2]

            # number of edits needed to get to target word
            dist = Lev.distance(target, word)
            distNoCap = Lev.distance(target.lower(), word.lower())
            # accuracy
            acc = get_accuracy(dist, target, word)
            accNoCap = get_accuracy(distNoCap, target, word)
            accY.append(acc)
            accNoCapY.append(accNoCap)
            # update average accuracies
            avgAc += acc
            avgAcNoCap += accNoCap
            avgAccY.append(avgAc/total)
            avgAccNoCapY.append(avgAc / total)
            if word != "":
                totalDetect += 1
                avgDetY.append(avgAc/totalDetect)
                avgDetNoCapY.append(avgAc / totalDetect)
        i += 1
    gtFile.close()

    # plotting graphs
    x1 = list(range(1, total+1))
    plt.scatter(x1, accY, label="Case Sensitive", s=1, color="orange")
    plot_labels("Image Number", "Percent Accuracy", "Accuracy of Each Image")
    plt.scatter(x1, accNoCapY, label="Ignore Case", s=1)
    plot_labels("Image Number", "Percent Accuracy", "Accuracy of Each Image")

    plt.scatter(x1, avgAccY, label="Case Sensitive", s=1, color="orange")
    plot_labels("Image Number", "Average Percent Accuracy", "Average Accuracy Across Images")
    plt.scatter(x1, avgAccNoCapY, label="Ignore Case", s=1)
    plot_labels("Image Number", "Average Percent Accuracy", "Average Accuracy Across Images")

    x2 = list(range(1, totalDetect + 1))
    plt.scatter(x2, avgDetY, label="Case Sensitive", s=1, color="orange")
    plot_labels("Image Number", "Average Percent Accuracy ", "Average Accuracy Across Images : Detected Words")
    plt.scatter(x2, avgDetNoCapY, label="Ignore Case", s=1)
    plot_labels("Image Number", "Average Percent Accuracy", "Average Accuracy Across Images : Detected Words")

    # final values
    print("Total Images: " + str(total) + "\tTotal Detected: " + str(totalDetect) + "\tTotal Undetected: " + str(total - totalDetect))
    print("Case Sensitive:\t" + str(avgAc / total) + "\tOnly Detected: " + str(avgAc / totalDetect))
    print("Ignore Case:\t" + str(avgAcNoCap / total) + "\tOnly Detected: " + str(avgAcNoCap / totalDetect))



def get_accuracy(dist, target, word):
    totalChar = len(word) if len(word) >= len(target) else len(target)
    if totalChar == 0: return 1
    correct = totalChar - dist
    return correct / totalChar

def plot_labels(x, y, title):
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    main()
