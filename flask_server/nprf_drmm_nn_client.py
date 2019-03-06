import requests
import numpy as np
import drmm_nn_client


class NN_client:

    def __init__(self, url):
        self.url = url
        self.headers = {'Accept': 'application/json', 'content-type': 'application/json'}

    def score_doc(self, q, d, topk_docs_text, topk_docs_scores):
        data = dict()
        data["query"] = q
        data["doc"] = d
        data["topk_docs_content"] = topk_docs_text
        data["topk_docs_score"] = topk_docs_scores
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "score", json=data, headers=self.headers)
        return float(response.text)

    # def score_doc_vec(self, q, d):
    #     data = dict()
    #     data["query"] = q
    #     data["doc"] = d
    #     self.headers['Connection'] = 'close'
    #     response = requests.post(self.url + "score_doc_vec", json=data, headers=self.headers)
    #     return float(response.text)

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
        response = requests.post(self.url + "transform_doc_inds_to_text", json=data, headers=self.headers).json()
        return response["doc"]

    # def fetch_background_data(self):
    #     self.headers['Connection'] = 'close'
    #     response = requests.get(self.url + "fetch_data", headers=self.headers).json()
    #     background_data = response["background_data"]
    #     print(type(background_data))
    #     print(len(background_data))
    #     print(np.asarray(background_data).shape)
    #     # print(background_data[0])
    #     return np.asarray(background_data)

    def fetch_model_input_data(self):
        self.headers['Connection'] = 'close'
        response = requests.get(self.url + "fetch_model_input_data", headers=self.headers).json()
        dd_q_feature = np.asarray(response["dd_q_feature"])
        dd_d_feature = np.asarray(response["dd_d_feature"])
        score_gate = np.asarray(response["score_gate"])
        print("dd_q_feat_shape", dd_q_feature.shape)
        print("dd_d_feat_shape", dd_d_feature.shape)
        print("score_gate_shape", score_gate.shape)
        return dd_q_feature, dd_d_feature, score_gate

    # def fetch_background_bm25_input_data(self, query):
    #     data = dict()
    #     data["query"] = query
    #     self.headers['Connection'] = 'close'
    #     response = requests.post(self.url + "fetch_background_bm25_data", json=data, headers=self.headers).json()
    #     query = response["query"]
    #     doc = response["doc"]
    #     print(np.asarray(query).shape)
    #     print(np.asarray(doc).shape)
    #     # print(background_data[0])
    #     return np.asarray(query), np.asarray(doc)

    def prepare_test_input_data(self, query, doc, topk_docs_text, topk_docs_scores):
        data = dict()
        data["query"] = query
        data["doc"] = doc
        data["top_nb_doc_list"] = topk_docs_text
        data["top_nb_doc_scores"] = topk_docs_scores
        self.headers['Connection'] = 'close'
        response = requests.post(self.url + "prepare_test_input", json=data, headers=self.headers).json()
        dd_q_feature = np.asarray(response["dd_q_feature"])
        dd_d_feature = np.asarray(response["dd_d_feature"])
        score_gate = np.asarray(response["score_gate"])
        sup_doc_topk_terms = np.asarray(response["sup_doc_topk_terms"])
        print(dd_q_feature.shape)
        print(dd_d_feature.shape)
        print(score_gate.shape)
        print(sup_doc_topk_terms.shape)
        return dd_q_feature, dd_d_feature, score_gate, sup_doc_topk_terms


if __name__ == '__main__':
    cli = NN_client("http://127.0.0.1:5010/")
    drmm_cli = drmm_nn_client.NN_client("http://127.0.0.1:5007/")
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
    doc_cont = drmm_cli.get_doc_content("FBIS3-10082")

    dd_q_feat, dd_d_feat, score_gate = cli.fetch_model_input_data()

    docs_text, docs_score = drmm_cli.get_bm25_matched_docs(query)
    # dd_q_feat, dd_d_feat, score_gate, sup_doc_topk_t = cli.prepare_test_input_data("international organized crime", doc_cont, docs_text, docs_score)

    # doc_vec = cli.transform_doc(doc_cont)

    # print(len(doc_vec))
    # print(doc_vec.count(106662))
    # print(doc_vec)
    # print(doc_cont)
    # print(d_inds)
    # doc_vec = ['0'] * len(d_inds[0].tolist())
    # doc_vec_text = cli.transform_doc_vec(d_inds[0].tolist())
    # doc_vec[:len(doc_vec_text)] = doc_vec_text
    # print(doc_vec)
    # print(len(doc_cont.split()))

    print(cli.score_doc("international organized crime", doc_cont, docs_text, docs_score))
    # print(cli.score_doc_vec("international organized crime", doc_vec))