# save this as app.py
import pandas as pd
import numpy as np
from flask import Flask, Response, request, render_template, flash, jsonify
from werkzeug.exceptions import HTTPException
from pathlib import Path
import comet.utils.dfcomp as dc 
#from flask_bootstrap import Bootstrap
from dfs.mydfs import *

pd.set_option('precision', 3)
from pathlib import Path
import pandas as pd
import click
from tqdm import tqdm
import pathlib
cur_path = pathlib.Path().resolve()
print("App started at ", cur_path)
myDfs = MyDfs()
app = Flask(__name__)

#Bootstrap(app)

#@app.errorhandler(Exception)
#def handle_exception(e):
#    # pass through HTTP errors
#    if isinstance(e, HTTPException):
#        return e
#
#    return render_template("err.html", msg=e), 500

import io
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

@app.route('/plot.png')
def plot_png():
    chart = request.args.get("chart", "na")
    fig = create_figure(chart)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

#ffff
def create_figure(chart):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    df = myDfs.cur["df"]
    tag = myDfs.cur["tag"]
    if "group" in chart:
        ax = df.pivot_table(index=[chart],columns=chart,aggfunc='size').plot(kind='bar')
    else:
        df = df[[chart]]
        ax = df.plot.hist()
    ax.set_title(chart)
    fig = ax.get_figure()
    r = random.randint(0, 1000)
    img = "images/" + tag + "_"  + chart + "_" + str(r) + ".png" 
    fig.savefig("static/" + img)
    key = tag + "_" + chart
    if key in myDfs.images:
        Path("static/" + myDfs.images[key]).unlink()
    myDfs.images[key] = img
    # axis.plot(xs, ys)
    return fig

metrics_list = ["bert score", "rouge", "bleu", "cider", "meteor"]

class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv["message"] = self.message
        return rv


@app.errorhandler(InvalidAPIUsage)
def invalid_api_usage(e):
    return render_template("err.html", msg=e.message)


import os

def format(item):
    if item is None:
        return item
    pred_class = "pred"
    if item["pred1_score"] == 1:
        pred_class="exact-match"
    elif item["pred1_score"] > 0.5:
        pred_class="top-score"

    if "top" in item:
        item["target_text"] = item["target_text"].replace(str(item["top"]), f'<span class="{pred_class}">' + str(item["top"]) + '</span>')
    item["pred_text1"] = f'<span class="{pred_class}">' + str(item["pred_text1"]) + '</span>'
    return item

def fill_opts(item, show_all = False):
    if item is None:
        return None
    titles = {
            "input_text":"Event", 
            "target_text": "Target", 
            "pred_text1":"Prediction",
            "pred1_score":"Prediction Score",
    }
    opts = {}
    for key,val in item.items():
        opt = {"show": show_all, "title": key, "class": "", "sort":False}
        if "group" in key:
            opt["group"] = True
        if not key in titles:
            titles[key] = key
        if "_fa" in key:
            opt["class"] += " persian"
        if "score" in key:
            opt["sort"] = True
            opt["filter"] = True
        if key in myDfs.show_list:
            opt["show"] = True
            opt["show_always"] = "checked"
            opt["title"] = titles[key]
        if key in myDfs.match_list:
            opt["match_always"] = "checked"
        opts[key] = opt
    return opts

@app.route("/", methods=["GET", "POST"])
def index():
    item_index = int(request.form.get("item_index", -1))
    cmd = request.args.get("cmd", "")
    show_list = request.form.getlist('show') 
    match_list = request.form.getlist('match') 
    if show_list:
        myDfs.show_list = show_list
    if match_list:
        myDfs.show_list = show_list
    if not cmd:
        cmd = request.form.get("cmd", "")
    cmd = cmd.strip().lower()
    cmd_target = ""
    if "@" in cmd:
        cmd, cmd_target = cmd.split("@")
    dfname = ""
    tag_dfname = request.form.get("tag_dfname", "")
    tag = tag_dfname
    report = ""
    if "=" in tag_dfname:
        tag, dfname = tag_dfname.split("=")
        tag = tag.strip()
        dfname = dfname.strip()
    elif tag in myDfs.dflist:
        dfname = myDfs.dflist[tag]
    else:
        tag = dfname = tag_dfname

    df = None
    #bbbbb
    if cmd == "make_hidden":
        myDfs.remove_entry(tag, from_orig_file=False)
        return ('', 204)
    if cmd == "remove":
        myDfs.remove_entry(tag, from_orig_file=True)
        return ('', 204)
    stag = tag.split("@")
    orig_tag = stag[0]
    if cmd == "orig_df":
        dfname = myDfs.dflist[orig_tag]
        report = ""
    if cmd == "rename":
        orig_tag = request.form.get("orig_tag", "")
        newtag = request.form.get("tag_name", "")
        if newtag != orig_tag and newtag in myDfs.dflist:
            return "Error: This name already exists!", 200
        myDfs.dflist[newtag] = myDfs.dflist[orig_tag]
        myDfs.remove_entry(orig_tag, from_file=True)
        return render_template(
            "dfentry.html",
            tag=newtag,
            path=myDfs.dflist[newtag],
        )
        
    if cmd == "reload":
        myDfs.reload(reset=True)
        return "", 204

    if not dfname:
        return render_template(
            "item.html",
            items=myDfs.items,
            dflist=myDfs.dflist,
        )
    # for filter or group commands, the data frame must reload
    if dfname in myDfs.dfs and not cmd in ["filter", "group"]:
        df = myDfs.dfs[dfname]["df"]
        myDfs.views[orig_tag] = {"dfname":dfname}
    else:
        df = myDfs.read_df(dfname)
        myDfs.views[orig_tag] = {"dfname":dfname}
        myDfs.dfs[dfname] = {"df":df, "tag":tag, "total":len(df), "item_index":0, "report":None}
        for col in df:
            if "group" in col:
              myDfs.groups[col] = df[col].unique()
        item_index = 0 
    if dfname:
        item_index = myDfs.dfs[dfname]["item_index"]

    item = None
    myDfs.scores[dfname] = {}
    myDfs.cur["tag"] = tag
    # ddddd
    show_report = False
    if cmd == "show_more":
        myDfs.show_all = True
    if cmd == "show_less":
        myDfs.show_all = False 
    if df is None:
        raise InvalidAPIUsage(dfname + " cannot be read")
    else:
        myDfs.cur["df"] = df
        if not "bookmark" in df:
            df["bookmark"] = False
            myDfs.save_df(df, dfname)
        df_folder = "/home/ahmad/dfs/" + tag.replace("@", "/")
        Path(df_folder).mkdir(exist_ok=True, parents=True)

        if cmd == "chart":
            return cmd_target, 203
        else:
            myDfs.images = {}
        if cmd == "group" or cmd_target in myDfs.groups:
            _group = request.form.get("group@" + cmd_target, "")
            df = df[(df[cmd_target] == _group)]

        if cmd == "filter" or cmd_target in myDfs.ranges.keys():
            score_range = request.form.get("range@" + cmd_target, "")
            if score_range:
                low,high = score_range.split("-")
                myDfs.ranges["pred1_score"]["low"] = float(low.strip())
                myDfs.ranges["pred1_score"]["high"] = float(high.strip())
            df = df[(df[cmd_target] <= myDfs.ranges[cmd_target]["high"]) & (df[cmd_target] >= myDfs.ranges[cmd_target]["low"])]
        if cmd == "sort" or (cmd_target in myDfs.sort and myDfs.sort[cmd_target]):
            df = df.sort_values(by=cmd_target, ascending=True)
        if cmd == "filter" or cmd == "sort":
            item_index = 0
            myDfs.sort[cmd_target] = True

        if cmd == "next":
            item_index += 1
        elif cmd == "previous":
            item_index -= 1
        elif cmd == "last":
            item_index = len(df) - 1

        if cmd == "set_baseline":
            myDfs.base["df"] = df
            myDfs.base["tag"] = tag 
            msg = "Baseline was set to " + tag
            return jsonify(message=msg), 201
        if cmd == "compare":
            mdfpath = df_folder + "/compare_" + myDfs.base["tag"] + "_" + tag + ".tsv"
            if myDfs.base["df"] is None:
                msg = "Please select the baseline dataframe to compare with"
                return jsonify(message=msg), 521
            if myDfs.base["df"].equals(df):
                msg = "Baseline should be a different dataframe"
                return jsonify(message=msg), 521
            elif Path(mdfpath).is_file():
                mdf = myDfs.read_df(mdfpath)
                myDfs.dfs[dfname]["report"] = mdf
            else:
                mdf = dc.compare(df_folder, myDfs.base["tag"], tag, myDfs.base["df"], df, cur_report=mdfpath)
                myDfs.save_df(mdf, mdfpath)
                myDfs.dfs[dfname]["report"] = mdf
            show_report = True

        myDfs.scores[dfname]["pred1_score"] = round(pd.to_numeric(df['pred1_score']).mean(),2)
        if "back_score" in df:
            myDfs.scores[dfname]["back_score"] = round(pd.to_numeric(df['back_score']).mean(),2)

        # mmmmmmm
        if cmd == "show_report":
            match_list = myDfs.match_list
            mdfpath = df_folder + "/" + "_".join(match_list) + "_report.tsv"
            if myDfs.dfs[dfname]["report"] is not None:
                mdf = myDfs.dfs[dfname]["report"]
            elif Path(mdfpath).is_file():
                mdf = myDfs.read_df(mdfpath, index_name="Report")
                myDfs.dfs[dfname]["report"] = mdf
            else:
                cmd = "reload_report"
            show_report =True
        if cmd == "reload_report" or cmd.strip() in metrics_list: 
            match_list = myDfs.match_list
            mdfpath = df_folder + "/" + "_".join(match_list) + "_report.tsv"
            metrics = {}
            if cmd in metrics_list:
                metrics[cmd] = True
                mdf = myDfs.read_df(mdfpath, index_name="Report")
            if cmd == "reload_report" and Path(mdfpath).is_file():
                Path(mdfpath).unlink()
            mdf = dc.make_report(df_folder, tag, df, metrics, match_list, cur_report=mdfpath)
            myDfs.dfs[dfname]["report"] = mdf
            myDfs.save_df(mdf, mdfpath)
            show_report =True

        if show_report:
            table=mdf.to_html(classes='table table-striped text-large', header=True, render_links=True, escape=False, na_rep="")
            table = table.replace('text-align: right;', '')
            return render_template('report.html', scores=myDfs.scores[dfname], tables=[table], titles=mdf.columns.values, metrics=metrics_list, tag=tag, dfname=dfname)

        item = myDfs.get_item(df, index=item_index)
        if item is None:
            msg = f"No item or match was found at {item_index}"
            return jsonify(message=msg, index=item_index, total=len(df)), 521

        item = format(item)

        if cmd == "bookmark":
            df.loc[item_index, 'bookmark'] = True
            item["bookmark"] = True
            myDfs.save_df(df, dfname)
        elif cmd == "unbookmark":
            df.loc[item_index, 'bookmark'] = False 
            item["bookmark"] = False
            myDfs.save_df(df, dfname)
            
        myDfs.dfs[dfname]["item_index"]=item_index
        myDfs.dfs[dfname]["df"]=df
        myDfs.dfs[dfname]["total"]=len(df)
        myDfs.items[orig_tag] = item
        if cmd == "unsync":
            myDfs.sync = False
        elif cmd == "sync" or myDfs.sync:
            myDfs.sync = True
            myDfs.match = item
            items = {}
            for key, _item in myDfs.items.items():
                _dfname = myDfs.views[key]["dfname"]
                _df = myDfs.dfs[_dfname]["df"]
                if _df is not None and not _df.equals(df):
                    sync_item = myDfs.get_item(_df, match = item)
                    ii = 0
                    if sync_item is not None:
                        ii = _df.index.get_loc(sync_item.name)
                    myDfs.dfs[_dfname]["item_index"] = ii
                    myDfs.items[key] = sync_item
        #eeeee
        items = {}
        for key, item in myDfs.items.items():
            _dfname = myDfs.views[key]["dfname"]
            _total = myDfs.dfs[_dfname]["total"]
            _tag = myDfs.dfs[_dfname]["tag"]
            _item_index = myDfs.dfs[_dfname]["item_index"]
            options = fill_opts(item, myDfs.show_all)
            item = format(item)
            df_scores = myDfs.scores[_dfname]
            items[key] = render_template(
                "row.html",
                item=item,
                images=myDfs.images,
                scores=df_scores,
                ranges=myDfs.ranges,
                groups=myDfs.groups,
                options = options,
                is_sync = myDfs.sync,
                show_all_records = myDfs.show_all,
                dfname = _dfname,
                tag = _tag,
                total = _total,
                dflist = myDfs.dflist,
                item_index = _item_index
            )
        return jsonify(items = items)


@app.route("/analysis/<filename>")
def analysis(filename):
    return render_template("base.html", name=filename, data=x.to_html())


#app.run(debug=True)


