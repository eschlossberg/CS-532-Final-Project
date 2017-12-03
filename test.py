import os

print(os.getcwd()+'/cleaned_articles')
top = os.getcwd()+'/cleaned_articles'

modals_path = "trained_modals"

try:
    os.makedirs(modals_path)
except OSError:
    # Prevent race condition
    if not os.path.isdir(modals_path):
        raise
os.chdir(modals_path)
print("Current wd after change is %s" % os.getcwd())
## get articles and put into doc2vec

for root, dirs, files in os.walk(top):
    for d in dirs:
        print("Now we are at: ")
        print(d)
        # Iterating companies in /cleaned_articles/
        sources = {}
        count = 1
        cwd = top + "/%s" % d
        for root, dirs, files in os.walk(cwd):
            for filename in files:
                print("File: %s"%filename)
                article_name_tag = d + "_%d" % count
                sources[filename] = article_name_tag
                count += 1

        print("Collected sources: ")
        print(sources)
    break
