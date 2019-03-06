import requests
import numpy as np


class NN_client:

    def __init__(self, url):
        self.url = url
        self.headers = {'Accept': 'application/json', 'content-type': 'application/json'}

    def score_doc(self, q, d):
        data = dict()
        data["query"] = q
        data["doc"] = d
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "score", json=data, headers=self.headers)
        return float(response.text)

    def score_doc_vec(self, q, d):
        data = dict()
        data["query"] = q
        data["doc"] = d
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "score_doc_vec", json=data, headers=self.headers)
        return float(response.text)

    def transform_doc(self, d):
        data = dict()
        data["doc"] = d
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "transform_doc", json=data, headers=self.headers).json()
        return response["doc_vec"]

    def transform_doc_vec(self, doc_vec):
        data = dict()
        data["doc_vec"] = doc_vec
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "transform_doc_vec", json=data, headers=self.headers).json()
        return response["doc"]

    def fetch_background_data(self):
        self.headers['Connection'] = 'close'
        response = requests.get(self.url + "fetch_data", headers=self.headers).json()
        background_data = response["background_data"]
        print(type(background_data))
        print(len(background_data))
        print(np.asarray(background_data).shape)
        # print(background_data[0])
        return np.asarray(background_data)

    def fetch_model_input_data(self):
        self.headers['Connection'] = 'close'
        response = requests.get(self.url + "fetch_model_input_data", headers=self.headers).json()
        query = response["query"]
        doc = response["doc"]
        print(np.asarray(query).shape)
        print(np.asarray(doc).shape)
        # print(background_data[0])
        return np.asarray(query), np.asarray(doc)

    def fetch_background_bm25_input_data(self, query):
        data = dict()
        data["query"] = query
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "fetch_background_bm25_data", json=data, headers=self.headers).json()
        query = response["query"]
        doc = response["doc"]
        print(np.asarray(query).shape)
        print(np.asarray(doc).shape)
        # print(background_data[0])
        return np.asarray(query), np.asarray(doc)

    def prepare_test_input_data(self, query, doc):
        data = dict()
        data["query"] = query
        data["doc"] = doc
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "prepare_test_input", json=data, headers=self.headers).json()
        query = response["query"]
        doc = response["doc"]
        q_hist_counts = response["q_hist_counts"]
        d_q_hist_map = response["d_q_hist_map"]
        print(np.asarray(query).shape)
        print(np.asarray(doc).shape)
        print(np.asarray(q_hist_counts).shape)
        # print(d_q_hist_map)
        return np.asarray(query), np.asarray(doc), np.asarray(q_hist_counts), d_q_hist_map

    def get_doc_content(self, doc_id):
        data = dict()
        data["doc_id"] = doc_id
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "doc_from_index", json=data, headers=self.headers).json()
        return response["doc_content"]

    def get_doc_bm25_score(self, query, doc_id):
        data = dict()
        data["query"] = query
        data["doc_id"] = doc_id
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "fetch_bm25_score", json=data, headers=self.headers)
        return float(response.text)

    def get_bm25_matched_docs(self, query, topk=1000, parse_text=True):
        data = dict()
        data["query"] = query
        data["k"] = topk
        data["parse_text"] = parse_text
        self.headers["Connection"] = "close"
        response = requests.post(self.url + "get_bm25_docs", json=data, headers=self.headers).json()
        dids = response["dids"]
        docs_text = response["docs_text"]
        scores = response["scores"]
        return dids, docs_text, scores

    def get_bg_docs_idf_ql(self, query):
        data = dict()
        data["query"] = query
        self.headers["Connection"] = "close"
        response = requests.post(self.url + "get_bg_docs_idf_ql", json=data, headers=self.headers).json()
        bg_doc_idf = response["doc_idf_low"]
        bg_doc_idf_ql_low = response["doc_idf_ql_low"]
        bg_doc_idf_ql_rand = response["doc_idf_ql_rand"]
        return bg_doc_idf, bg_doc_idf_ql_low, bg_doc_idf_ql_rand

    def get_bg_rand_collect_doc(self, query):
        data = dict()
        data["query"] = query
        self.headers["Connection"] = "close"
        response = requests.post(self.url + "get_rand_collect_doc", json=data, headers=self.headers).json()
        return response["rand_collect_doc"]

    def get_bg_rand_res_doc(self, query):
        data = dict()
        data["query"] = query
        self.headers["Connection"] = "close"
        response = requests.post(self.url + "get_rand_res_doc", json=data, headers=self.headers).json()
        return response["rand_res_doc"]


if __name__ == '__main__':
    cli = NN_client("http://127.0.0.1:5007/")
    # cli = drmm_client("http://130.75.152.42:5000/score")
    # print(cli.score_doc("meeting", "hi greetings nice to meet you"))

    # a = np.zeros((10,), dtype=np.int32)
    # a[2] = 2
    # a[5] = 1
    # a[0] = 1
    # a[1] = 1
    # print(cli.score_doc_vec("meeting", a.tolist()))

    # doc_vec = cli.transform_doc("hi greetings nice to meet you")
    # print(type(doc_vec))
    # print(len(doc_vec))
    # print(doc_vec)
    # print(doc_vec.count('0'))
    # print(cli.score_doc_vec("meeting", doc_vec))

    # cli.fetch_background_data()
    query = "international organized crime"
    # Q, D = cli.fetch_background_bm25_input_data(query)
    # Q, D = cli.fetch_model_input_data()
    # print(D[0])
    # print(D[1])
    doc_cont = cli.get_doc_content("FBIS3-10082")
    cli.prepare_test_input_data("international organized crime", doc_cont)
    # doc_vec = cli.transform_doc(doc_cont)
    print(cli.get_doc_bm25_score(query, "FBIS3-12450"))

    # print(len(doc_vec))
    # print(doc_vec.count(106662))
    # print(doc_vec)
    # print(doc_cont)
    # print(cli.transform_doc_vec(doc_vec))
    # print(len(doc_cont.split()))
    # print(cli.score_doc("international organized crime", doc_cont))
    # print(cli.score_doc_vec("international organized crime", doc_vec))
