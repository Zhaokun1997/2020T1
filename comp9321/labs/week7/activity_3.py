import requests


def print_book(book):
    print("Book {")
    for key in book.keys():
        attr = str(key)
        val = str(book[key])
        print("\t" + attr + ":" + val)
    print("}")


def get_book(id):
    resp_obj = requests.get("http://127.0.0.1:5000/books/" + str(id))
    print("respond is: ", resp_obj)
    book = resp_obj.json()
    print("Get status Code:" + str(resp_obj.status_code))
    if resp_obj.ok:
        print_book(book)
        return book
    else:
        print("Error: " + book['message'])


if __name__ == '__main__':
    print("***** Book information before update *****")
    book = get_book(206)

    # update the book information
    print("***** Updating Book Information *****")
    book['Author'] = 'Nobody'
    # book['Identifier'] = 206
    resp_obj = requests.put("http://127.0.0.1:5000/books/206", json=book)
    print("Put status Code:" + str(resp_obj.status_code))
    print(resp_obj.json()['message'])

    print("***** Book information after update *****")
    book = get_book(206)
