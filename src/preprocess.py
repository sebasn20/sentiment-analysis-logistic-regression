import re
def clean_text(text):
    # everything lowercase
    text = text.lower()

    # replace ampersand with and
    text = text.replace('&amp', 'and')

    # turn smiley/frowns into normal words
    text = text.replace(':)', 'smile')
    text = text.replace(':(', 'frown')

    # turn isnt and dont into is not and do not
    text = text.replace("n't", " not")

    # remove links and @ mentions
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    # remove # and anything that isnt a letter, number, or space and remove extra spaces
    text = text.replace("#", " ")
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = " ".join(text.split())

    return text


# Test it on a few example tweets
if __name__ == "__main__":
    examples = [
        "@AmericanAir the service isn't great. Losing luggage is bs",
        "@SouthwestAir thanks! I expected a wait... :)",
        "@united I can't believe this!!! #worstflight http://t.co/abc123",
    ]
    for tweet in examples:
        print("BEFORE:", tweet)
        print("AFTER: ", clean_text(tweet))
        print()