# 車刀拍照、推論、錄音系統

## Docker 服務

這版是 3 個 Docker 服務：

```text
8001 camera-web      前端網頁 + 拍照 / 推論 / 錄音流程
8002 detection-api   物件偵測 + bbox 1.5 倍裁切
8003 keypoint-api    關鍵點分析 + 刃口聚焦裁切
```

`8003` 的刃口聚焦裁切已加大範圍，會比原本多保留周圍一些畫面。

## 操作流程

1. 執行 `start_docker.bat`，這台電腦會固定開啟 `https://localhost:8001`。
2. 左側約 2/3 是即時相機畫面。
3. 按 `拍照`。
4. 右側約 1/3 會顯示剛拍的照片，確認有沒有聚焦。
5. 如果沒聚焦，可以再按 `拍照` 重拍。
6. 確認沒問題後按 `推論`。
7. 推論結果會依階段顯示：
   - 物件與關鍵點都成功：上方顯示 `分析完成`，右側顯示最後刃口成果圖。
   - 物件成功、關鍵點失敗：上方顯示 `關鍵點偵測失敗，請重新拍攝`，右側顯示只有 bbox 的物件偵測圖。
   - 物件偵測失敗：上方顯示 `物件偵測失敗，請重新拍攝`，右側顯示原圖。
8. 按 `開始錄音`，講完後按 `停止錄音`。
9. 錄音成功後，才會存檔並允許下一筆資料。

## 防呆規則

- 沒拍照不能推論。
- 沒推論不能錄音。
- 錄音中不能拍照或推論。
- 錄音成功存檔後，才會開始下一筆資料。
- 任一推論階段失敗後必須重新拍照。

## 輸出

`photos` 會存同名的最終推論照片與 MP4 錄音：

```text
photos/tool_時間戳.jpg
photos/tool_時間戳.mp4
```

`datasets` 會分成兩類：

```text
datasets/detection/tool_時間戳_object_detection.jpg
datasets/keypoint/tool_時間戳_keypoint_detection.jpg
```

- `datasets/detection`：物件偵測那一步的帶框原圖，不裁切、不放大、不塗色。
- `datasets/keypoint`：最後聚焦刃口的關鍵點推論成果。

`results` 是詳細中間結果與 JSON，平常可以不用看。

## 啟動

1. 開啟 Docker Desktop。
2. 等待 Docker Engine running。
3. 雙擊 `start_docker.bat`。
4. 啟動程式會自動偵測目前區網 IPv4，建立並信任本機 HTTPS 憑證。
5. 這台電腦的瀏覽器會開啟 `https://localhost:8001`，不需要手動填 IP。
6. 黑色視窗也會列出其他裝置可用的區網網址，例如 `https://192.168.1.20:8001`。

若老師要求 `192.168.x.x`，請先讓這台電腦連上老師指定的路由器，再重新執行
`start_docker.bat`。IP 是由路由器分配，程式不能把手機熱點的
`172.20.10.x` 直接改成 `192.168.x.x`。

啟動檔會使用 `--remove-orphans`，自動清掉舊版不再使用的容器，例如舊的 `crop-api`。

## 關閉

雙擊 `stop_docker.bat`。
