# **開放平台Final-Project**

**Jupyter website:** http://140.138.144.130:30577/

## **須完成項目:**

* [ ] Dataset
* [ ] data_loader.py
* [ ] training.py
* [ ] test.py
* [ ] ppt or pdf - 講述code的實作 (Presetation)
* [ ] pdf - 企劃書 內容可能包括動機 市場 UI操作流程等 (SRS)
* [ ] UI

## **Git 實用指令:**
* `git clone https://github.com/JpyChiu/DeepLearning-Final.git`
把專案(包括git相關設定download到自己電腦)
* `git pull`
把現在github上的檔案與自己電腦裡的專案做更新
* `git add .`
把目前有更改的所有檔案加入暫存區
* `git commit -m "commit meesage"`
把暫存區的檔案commit
* `git push`
把commit push到github上(第一次push需要設定push位置 ex: `git push -u origin master`)
* `git status`
查看目前自己改動了哪些檔案(紅字為尚未存入暫存區 綠字為已加入暫存區)
* `git log`
查看commit log訊息

## **Latex 安裝**
* 下載 `Miktex` 安裝包 `https://miktex.org/download`
* 可以使用安裝包內的 `TeXworks` 編輯器編譯 `.tex` 檔
* 編譯後會出現除了 `.pdf` ，還會有多個小檔案，如:  `.aux`， `.log`， `.syntex.gz` 等檔案
  * 已經在 `.gitignore` 裡面已排除這些檔案, 不會上傳到github上