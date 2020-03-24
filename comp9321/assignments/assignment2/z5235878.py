import json
import sqlite3
import pandas as pd
from pandas.io import sql
from flask import Flask
from flask import request
from flask_restplus import Resource, Api
from flask_restplus import fields
from flask_restplus import reqparse
from flask_restplus import inputs
import requests
import datetime
import os
import re

app = Flask(__name__)
api = Api(app,
          default="World Bank Economic Indicators",
          title="economic indicator data for countries",
          description="allows a client to read and store some publicly\
           available economic indicator data for countries around the world, \
           and allow the consumers to access the data through a REST API.")


# database execute sql script
def Database_Excute_Command(db_file, command):
    # create a connection
    cnx = sqlite3.connect(db_file)
    # create a cursor
    cursor = cnx.cursor()

    # if there is one time ';' : read database
    # if there are multiple ';' : create database
    if command.count(';') > 1:
        cursor.executescript(command)
    else:
        cursor.execute(command)
    # when create database, no output fetched
    returnValue = cursor.fetchall()
    cnx.commit()
    cnx.close()
    return returnValue


# Initialise sqlite database
def Init_Database(db_file):
    # database path already exists
    if os.path.exists(db_file):
        print("Database already exists!")
        return
    print("Creating database...")
    command = 'CREATE TABLE Collection ( ' \
              'id INTEGER UNIQUE NOT NULL, ' \
              'indicator_id VARCHAR(50), ' \
              'indicator_value VARCHAR(50), ' \
              'uri VARCHAR(50), ' \
              'creation_time DATE, ' \
              'CONSTRAINT collection_pkey PRIMARY KEY (id) ); ' \
              + \
              'CREATE TABLE Entries ( ' \
              'id INTEGER NOT NULL, ' \
              'country VARCHAR(50), ' \
              'date VARCHAR(50), ' \
              'value VARCHAR(50), ' \
              'CONSTRAINT entries_fkey FOREIGN KEY (id) REFERENCES Collection(id) ); '

    return Database_Excute_Command(db_file, command)


# post a requested data record into the database
def post_executor(db_file, indicator_id):
    fetched_rows = Database_Excute_Command(db_file, f'SELECT * FROM Collection WHERE indicator_id = "{indicator_id}";')
    # if there already exists
    if fetched_rows:
        return {"uri": "/collections/{}".format(fetched_rows[0][0]),
                "id": fetched_rows[0][0],
                "creation_time": "{}".format(fetched_rows[0][4]),
                "indicator_id": "{}".format(fetched_rows[0][1])}, 200
    else:
        resp = requests.get("http://api.worldbank.org/v2/countries/all/indicators/" \
                            + indicator_id + \
                            "?date=2012:2017&format=json&per_page=1000")
        data = resp.json()
        # with open("111.json", 'r') as load_file:
        #     data = json.loads(load_file)

        if re.findall('Invalid value', str(data), flags=re.I) or not data:
            return {"message": "Indicator_id cannot find in the resource"}, 404
        else:  # if remote source is avaliable
            new_id = re.findall('\d+', str(Database_Excute_Command(db_file, 'SELECT MAX(id) FROM Collection;')))
            if not new_id:
                new_id = 1
            else:
                new_id = int(new_id[0]) + 1
            # insert new data into database
            # get current time
            created_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H:%M:%SZ')
            insert_collection = "INSERT INTO Collection VALUES({}, '{}', '{}', '{}', '{}');" \
                .format(new_id, indicator_id, data[1][0]['indicator']['id'], data[1][0]['indicator']['value'],
                        created_time)
            Database_Excute_Command(db_file, insert_collection)

            insert_entries = "INSERT INTO Entries VALUES "
            for sub_item in data[1]:
                insert_entries += f"( {new_id}, '{sub_item['country']['value']}', '{sub_item['date']}', " \
                                  f"'{sub_item['value']}' ),"
            insert_entries = insert_entries.rstrip(',') + ';'
            Database_Excute_Command(db_file, insert_entries)

            select_result = Database_Excute_Command(db_file,
                                                    f"SELECT * FROM Collection WHERE indicator_id = '{indicator_id}'; ")
            return {"uri": "/collections/{}".format(select_result[0][0]),
                    "id": select_result[0][0],
                    "creation_time": "{}".format(select_result[0][4]),
                    "indicator_id": "{}".format(select_result[0][1])}, 201


# delete a given record from the database
def delete_executor(db_file, id):
    fetch_rows = Database_Excute_Command(db_file, f"SELECT * FROM Collection WHERE id = {id};")
    if not fetch_rows:
        return {"message": f"Index {id} not found in the database!", "id": id}, 404
    else:
        Database_Excute_Command(db_file, f"DELETE FROM Entries WHERE id = {id}; ")
        Database_Excute_Command(db_file, f"DELETE FROM Collection WHERE id = {id}; ")
        return {"message": f"The collection {id} was removed from the database!", "id": id}, 200


# get all unsorted records from the database
def getAll_executor(db_file):
    fetch_rows = Database_Excute_Command(db_file, f"SELECT * FROM Collection;")
    if not fetch_rows:
        return {"message": f"No records found in the database!"}, 404
    else:
        returnList = []
        for i in range(len(fetch_rows)):
            returnList.append({"uri": "/collections/{}".format(fetch_rows[i][0]),
                               "id": fetch_rows[i][0],
                               "creation_time": "{}".format(fetch_rows[i][4]),
                               "indicator_id": "{}".format(fetch_rows[i][1])})
        return returnList


# get one record with required content from the database
def getOne_executor(db_file, id):
    fetch_rows = Database_Excute_Command(db_file, f"SELECT * FROM Collection WHERE id = {id};")
    if not fetch_rows:
        return {"message": f"Index {id} not found in the database!", "id": id}, 404
    else:
        fetch_collections = fetch_rows
        fetch_entries = Database_Excute_Command(db_file, f"SELECT * FROM Entries WHERE id = {id};")
        country_list = []

        # format: [{"country": "Australia", "date": 2016, "value": 780016444034.00}, ...]
        for i in range(len(fetch_entries)):
            country_list.append(
                {
                    "country": "{}".format(fetch_entries[i][1]),
                    "date": "{}".format(fetch_entries[i][2]),
                    "value": "{}".format(fetch_entries[i][3])
                }
            )
        returnValue = {"id": fetch_collections[0][0],
                       "indicator_id": "{}".format(fetch_collections[0][1]),
                       "indicator_value": "{}".format(fetch_collections[0][2]),
                       "creation_time": "{}".format(fetch_collections[0][4]),
                       "entries": country_list
                       }
        return returnValue, 200


# get one record with required triple parameters from the database
def getOneWithTripleParams_executor(db_file, id, year, country):
    fetch_rows = Database_Excute_Command(db_file, f"SELECT * FROM Collection WHERE id = {id};")
    if not fetch_rows:
        return {"message": f"Index {id} not found in the database!", "id": id}, 404
    else:
        fetch_collection = fetch_rows
        fetch_entry = Database_Excute_Command(db_file, f"SELECT * FROM Entries "
                                                       f"WHERE id = {id} AND "
                                                       f"date = '{year}' AND "
                                                       f"country = '{country}';")
        returnValue = {
            "id": fetch_collection[0][0],
            "indicator_id": "{}".format(fetch_collection[0][1]),
            "country": "{}".format(fetch_entry[0][1]),
            "year": "{}".format(fetch_entry[0][2]),
            "value": "{}".format(fetch_entry[0][3])
        }
        return returnValue, 200


# get a collection of top/bottom economic indicator values for a given year
def getTopOrBottom_executor(db_file, id, year, query):
    fetch_rows = Database_Excute_Command(db_file, f"SELECT * FROM Collection WHERE id = {id};")
    if not fetch_rows:
        return {"message": f"Index {id} not found in the database!", "id": id}, 404
    else:
        fetch_collection = fetch_rows
        fetch_entries = Database_Excute_Command(db_file, f"SELECT * FROM Entries "
                                                         f"WHERE id = {id} AND "
                                                         f"date = '{year}';")
        country_list = []
        for i in range(len(fetch_entries)):
            if fetch_entries[i][3] == 'None':
                country_list.append(
                    {
                        "country": "{}".format(fetch_entries[i][1]),
                        "value": 0.0
                    }
                )
            else:
                country_list.append(
                    {
                        "country": "{}".format(fetch_entries[i][1]),
                        "value": float(fetch_entries[i][3])
                    }
                )
        # start to sort
        nb_country = int(query[1:])  # nb of display countries
        side = query[0]  # top or bottom

        # higher value first --> descending order
        sorted_country_list = sorted(country_list, key=lambda x: x['value'], reverse=True)
        if side == '+':
            returnValue = {
                "indicator_id": "{}".format(fetch_collection[0][1]),
                "indicator_value": "{}".format(fetch_collection[0][2]),
                "entries": sorted_country_list[0:nb_country]
            }
            return returnValue, 200
        else:
            returnValue = {
                "indicator_id": "{}".format(fetch_collection[0][1]),
                "indicator_value": "{}".format(fetch_collection[0][2]),
                "entries": sorted_country_list[len(sorted_country_list) - nb_country - 1:len(sorted_country_list)]
            }
            return returnValue, 200


def request_executor(db_file, request_type, **kwargs):
    if request_type == 'post':
        return post_executor(db_file, kwargs['indicator_id'])
    elif request_type == 'delete':
        return delete_executor(db_file, kwargs['id'])
    elif request_type == 'getAll':
        return getAll_executor(db_file)
    elif request_type == 'getOne':
        return getOne_executor(db_file, kwargs['id'])
    elif request_type == 'getOneWithTripleParams':
        return getOneWithTripleParams_executor(db_file, kwargs['id'], kwargs['year'], kwargs['country'])
    elif request_type == 'getTopOrBottom':
        return getTopOrBottom_executor(db_file, kwargs['id'], kwargs['year'], kwargs['query'])
    else:
        return {"message": "There is no such request type!"}, 400


# post_model = api.model('Post_payload', {'indicator_id': fields.String})
# getAll_model = api.model('GetAll_payload', {'order_by': fields.String})
parser = reqparse.RequestParser()
parser.add_argument('indicator_id', type=str, help='Add your indicator_id here', location='args')
parser.add_argument('order_by', type=str, help='Add a criteria to order results', location='args')
parser.add_argument('query', type=str, help='Add a critera to order countries', location='args')


# all Api
@api.route('/collections')
class FirstAndThirdRoute(Resource):
    # Question 1:
    @api.response(200, "Indicator_id Already Exists")
    @api.response(201, "Created")
    @api.response(400, "Request Type Not Match")
    @api.response(404, "Not Found")
    # @api.param('indicator_id', "The Indicator ID")
    @api.doc(params={'indicator_id': 'the indicator id'}, description="Add an indicator collection")
    def post(self):
        indicator_id = parser.parse_args()['indicator_id']
        # print(indicator_id)
        if not indicator_id:
            return {"message": "Please give your query indicator_id!"}, 400
        else:
            return request_executor('z5235878.db', 'post', indicator_id=indicator_id)

    # Question 3:
    @api.response(200, "OK")
    @api.response(400, "Request Type Not Match")
    @api.response(404, "Not Found")
    # @api.param('order_by', "the ordering criteria")
    @api.doc(params={'order_by': "the ordering criteria"}, description="Get all collections with given order")
    def get(self):
        query = parser.parse_args()['order_by']
        if not query:
            return {"message": "Please give your ideal order!"}, 400
        else:
            # first fetch all data that has not been sorted
            unsort_result = request_executor('z5235878.db', 'getAll')  # a list with json dict

            # start to sort
            order_critera = query.split(',')
            # sample: +id,+creation_time,+indicator
            # , and
            for item in order_critera:
                if item[0] == '+':  # ascending order
                    # + ascending order
                    unsort_result = sorted(unsort_result, key=lambda x: x[item[1:]], reverse=False)
                else:
                    # - descending order
                    unsort_result = sorted(unsort_result, key=lambda x: x[item[1:]], reverse=True)
            return unsort_result, 200


@api.route('/collections/<int:id>')
class SecondAndFourthRoute(Resource):
    # Question 2:
    @api.response(200, "OK")
    @api.response(400, "Request Type Not Match")
    @api.response(404, "Not Found")
    @api.doc(description="Delete a required collection from database")
    def delete(self, id):
        if not id:
            return {"message": "Please give your query id!"}, 400
        else:
            return request_executor('z5235878.db', 'delete', id=id)

    # Question 4:
    @api.response(200, "OK")
    @api.response(400, "Request Type Not Match")
    @api.response(404, "Not Found")
    @api.doc(description="Get a collection with required content from database")
    def get(self, id):
        if not id:
            return {"message": "Please give your query id!"}, 400
        else:
            return request_executor('z5235878.db', 'getOne', id=id)


@api.route('/collections/<int:id>/<int:year>/<string:country>')
class FifthRoute(Resource):
    # Question 5:
    @api.response(200, "OK")
    @api.response(400, "Request Type Not Match")
    @api.response(404, "Not Found")
    @api.doc(description="Get a collection with specific (id, year and country) from database")
    def get(self, id, year, country):
        if not id or not year or not country:
            return {"message": "Please give your query id!"}, 400
        else:
            return request_executor('z5235878.db', 'getOneWithTripleParams', id=id, year=year, country=country)


@api.route('/collections/<int:id>/<int:year>')
class SixthRoute(Resource):
    # Question 6:
    @api.response(200, "OK")
    @api.response(400, "Request Type Not Match")
    @api.response(404, "Not Found")
    # @api.param('query': "the ordering criteria for country")
    @api.doc(params={'query': "the ordering criteria for country"},
             description="Get a collection of top/bottom economic indicator values for a given year")
    def get(self, id, year):
        query = parser.parse_args()['query']
        if not query or not id or not year:
            return {"message": "Please check your given parameters!"}, 400
        else:
            if 1 <= int(query[1:]) <= 100:
                return request_executor('z5235878.db', 'getTopOrBottom', id=id, year=year, query=query)
            else:
                return {"message": "Please check the range of 'query' parameter!"}, 400


if __name__ == '__main__':
    database_file = 'z5235878.db'
    res = Init_Database(database_file)  # created successfully
    # print("Database created successfully!", res)

    # requestlist = "1.0.HCount.1.90usd", "1.0.HCount.Mid10to50", "1.0.HCount.Ofcl"
    app.run(debug=True)
