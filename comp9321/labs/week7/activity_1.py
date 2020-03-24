import requests


def print_book(book):
    print("Book {")
    for key in book.keys():
        attr = str(key)
        # if isinstance(book[key], str):
        #     val = str(book[key].encode('utf-8'))
        # else:
        val = str(book[key])

        print("\t" + attr + ":" + val)
    print("}")


if __name__ == '__main__':
    resp = requests.get("http://127.0.0.1:5000/books", params={'order': "Date_of_Publication", 'ascending': True})
    print("Status code: ", str(resp.status_code))
    books = resp.json()
    # print(books)
    for i in range(1, 5):
        print_book(books[i])
