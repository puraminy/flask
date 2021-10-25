import shutil
import json
import pandas as pd
from pathlib import Path, PurePath
import os
import os.path

class MyDfs():
    def __init__(self):
        self.reload()

    def reload(self, reset=False):
        self.items = {}
        self.views = {}
        self.match = {"prefix":"", "input_text":"", "target_text":""}
        self.dfs = {}
        self.images = {}
        self.cur = {"df":None, "tag":""}
        self.sort = {"pred1_score": True}
        self.scores = {}
        self.show_all = False
        self.base = {"df":None, "tag":""}
        self.sync = False
        self.ranges = {
                "pred1_score":{"low":0, "high":1},
                "back_score":{"low":0, "high":1}
        }
        self.groups = {}
        self.dflist = {}
        self.show_list = ["input_text", "target_text", "pred_text1", "pred1_score"]
        self.match_list = ["pred_text1", "target_text"]
        self.dflist_orig_file = "/home/pouramini/dflist"
        self.dflist_temp_file = "/home/pouramini/dflist-temp"
        self.src_dfname = "/drive3/pouramini/data/atomic/en_fa/xIntent_en_fa_de_train_no_dups.tsv"
        if not Path(self.dflist_temp_file).exists() or reset:
           shutil.copy(self.dflist_orig_file, self.dflist_temp_file)
        with open(self.dflist_temp_file) as f:
            lines = f.readlines()
        for l in lines:
            tag, dfname = l.split("=")
            tag = tag.replace(".", "_")
            if Path(dfname.strip()).is_file():
                self.dflist[tag.strip()] = dfname.strip()
        #self.get_files()
        if False:
            for tag, dfname in self.dflist.items():
                df1 = pd.read_table(dfname, index_col=0)
                df2 = pd.read_table(self.src_dfname, index_col=0)
                mdf = pd.merge(df1, df2)
                mdf.to_csv(dfname, sep="\t", index=False)
        with open('/home/pouramini/dflist.json', 'w') as fp:
                json.dump(self.dflist, fp)


    def get_files(self):
        cur_path = Path().resolve()
        for dirpath, dirnames, filenames in os.walk("."):
            for filename in [f for f in filenames if "scored_" in f]:
                dfname = PurePath(cur_path, dirpath, filename)
                dfname = str(dfname)
                tag = filename
                tag = tag.replace(".", "_")
                print("dfname:", dfname)
                self.dflist[tag.strip()] = dfname.strip()

    def remove_entry(self, tag, from_orig_file = False):
        del self.dflist[tag]
        self.save_dflist()
        if from_orig_file:
            self.save_dflist(orig=True)

    def save_dflist(self, orig =False):
        if orig:
            _file = self.dflist_orig_file
        else:
            _file = self.dflist_temp_file
        with open(_file, "w") as f:
            for k,v in self.dflist.items():
                print(f"{k.strip()}={v.strip()}", file=f)

    def read_df_by_tag(self, tag):
        dfname = self.dflist[tag]
        return self.read_df(dfname)

    def read_df_by_index(self, index):
        dfname = list(self.dflist.values())[index]
        return self.read_df(dfname)

    def read_df(self, dfname, index_name =""):
        dfname= dfname.strip()
        if not dfname or not Path(dfname).exists():
            return None

        if dfname.endswith("csv"):
            df = pd.read_csv(dfname, index_col=0)
        elif dfname.endswith("tsv"):
            df = pd.read_table(dfname, index_col=0)
            print("reading df:",len(df)) 
        #df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
        if index_name:
            df.columns.name = index_name
        return df

    def save_df(self, df, dfname, ind = True):
        dfname= dfname.strip()
        if dfname.endswith("csv"):
            df.to_csv(dfname, index=ind)
        elif dfname.endswith("tsv"):
            df.to_csv(dfname, sep="\t", index=ind)

    natural={
       "xIntent":"PersonX intends"
    }
    def match(df, col1, col2):
        return df[df.apply(lambda x: x[col1] in x[col2], axis=1)]

    def get_item(self, df, index=0, match=None):
        item = None
        if index >= 0 and index < len(df):
            item = df.iloc[index]
        elif match is not None:
            matches = df[(df['input_text'] == match['input_text']) 
                 & (df['prefix'] == match['prefix'])]
                 # & (df['target_text'] == item['target_text'])].iloc[0]
            if matches is not None and len(matches) > 0:
                item = matches.iloc[0]
        if item is not None:
            if "input_text_fa" in df:
                input_text_fa = item.input_text_fa.replace("PersonX"," شخص الف")
                item.input_text_fa = input_text_fa
            if "prefix" in df:
                item["relation"] = self.natural[item.prefix]
        return item

