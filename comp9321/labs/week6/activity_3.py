import json
import pandas as pd
from flask import Flask
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import fields
from flask_restplus import reqparse
from flask_restplus import inputs

app = Flask(__name__)
api = Api(app,
          default="Books",  # Default namespace
          title="Book Dataset",  # Documentation Title
          description="This is a simple sample to show how publish data as a service")  # Documentation Description

# The following is the schema of Book
book_model = api.model('Book', {
    'Flickr_URL': fields.String,
    'Publisher': fields.String,
    'Author': fields.String,
    'Title': fields.String,
    'Date_of_Publication': fields.Integer,
    'Identifier': fields.Integer,
    'Place_of_Publication': fields.String
})

parser = reqparse.RequestParser()
parser.add_argument('order', choices=list(column for column in book_model.keys()))
parser.add_argument('ascending', type=inputs.boolean)


@api.route('/books')
class BooksList(Resource):

    @api.response(200, "Successful")
    @api.doc(description="Get all books")
    def get(self):
        # get books as JSON string
        args = parser.parse_args()

        order_by = args.get('order')
        ascending = args.get('ascending', True)

        if order_by:
            df.sort_values(by=order_by, ascending=ascending, inplace=True)

        # output as a string JSON that is of form : 'index：{columns：values}'
        json_str = df.to_json(orient='index')

        # convert the string JSON to a real JSON
        ds = json.loads(json_str)
        returnList = []
        for idx in ds:
            book = ds[idx]
            book['Identifier'] = int(idx)
            returnList.append(book)

        return returnList

    # to validate if the parameter is valid or not(if not raise an error)
    @api.response(201, "Book created successfully")
    @api.response(400, "Validation Error")
    @api.doc(description="Add a new book")
    @api.expect(book_model, validate=True)
    def post(self):
        book = request.json

        if 'Identifier' not in book:
            return {"message": "Missing Identifier"}, 400

        id = book['Identifier']

        # check if the given identifier exists in the server
        if id in df.index:
            return {"message": "A book with Identifier{} is already in the dataset".format(id)}, 400

        # if not exist, then add it
        # Put the values into the dataframe
        for key in book:
            # unexpected column
            if key not in book_model.keys():
                return {"message": "Property {} is invalid".format(key)}, 400
            df.loc[id, key] = book[key]
        # df2 = df.append(book, ignore_index=True)
        # print(df2)
        return {"message": "Book {} is created".format(id)}, 201


@api.route('/books/<int:id>')
@api.param('id', "The Book identifier")
class Books(Resource):
    @api.response(404, "Book was not found")
    @api.response(200, "Successful")
    @api.doc(description="Get a book by its ID")
    def get(self, id):
        if id not in df.index:
            api.abort(404, "Book {} does not exist".format(id))
        # else
        book = dict(df.loc[id])
        return book

    @api.response(404, "Book was not found")
    @api.response(200, "Successful")
    @api.doc(description="Delete a book by its ID")
    def delete(self, id):
        if id not in df.index:
            api.abort(404, "Book {} does not exist".format(id))
        # else
        df.drop(index=id, inplace=True)
        return {"message": "Book {} is removed.".format(id)}, 200

    @api.response(404, "Book was not found")
    @api.response(400, "Validation Error")
    @api.response(200, "Successful")
    @api.doc(description="Update a book by its ID")
    @api.expect(book_model)  # expected input form
    def put(self, id):
        if id not in df.index:
            api.abort(404, "Book {} does not exist".format(id))

        # get the payload and convert it to a JSON
        book = request.json

        # if id of the book you want to change is different from provided book
        # Book ID cannot be changed
        if 'Identifier' in book and id != book['Identifier']:
            return {"message": "Identifier cannot be changed".format(id)}, 400

        # if there is some unconsistences
        for key in book:
            # unexpected columns
            if key not in book_model.keys():
                return {"message": "Property {} is invalid".format(key)}, 400

        # everything standard: update the values
        for key in book:
            df.loc[id, key] = book[key]
        return {"message": "Book {} has been successfully updated".format(id)}, 200


if __name__ == '__main__':
    columns_to_drop = ['Edition Statement',
                       'Corporate Author',
                       'Corporate Contributors',
                       'Former owner',
                       'Engraver',
                       'Contributors',
                       'Issuance type',
                       'Shelfmarks'
                       ]
    csv_file = 'Books.csv'
    df = pd.read_csv(csv_file)

    # drop unneccessary columns
    df.drop(columns_to_drop, inplace=True, axis=1)

    # clean the date of publication & convert it to numeric data
    new_date = df['Date of Publication'].str.extract(r'^(\d{4})', expand=False)
    new_date = pd.to_numeric(new_date)
    new_date = new_date.fillna(0)
    df['Date of Publication'] = new_date

    # replace spaces in the name of columns
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # set the index column; this will help us to find books with their ids
    df.set_index('Identifier', inplace=True)

    # run the application
    app.run(debug=True)
