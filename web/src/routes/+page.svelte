<!-- routes/+page.svelte.svelte -->
<script>
  let selectedImage;
  let restaurantData;
  let editedData;
  let serverResponse;
  let uploading = false;
  let showScreenA = true;

  const handleImageChange = (event) => {
      const file = event.target.files[0];
      selectedImage = file;
  };

  const handleUpload = async () => {
      uploading = true;
      const formData = new FormData();
      formData.append("image", selectedImage);

      try {
          const response = await fetch("/api/menu", {
              method: "POST",
              body: formData,
          });

          if (response.ok) {
              serverResponse = await response.json();
              restaurantData = serverResponse;
              editedData = { ...restaurantData };
          } else {
              serverResponse = { error: "Upload failed." };
          }
      } catch (error) {
          serverResponse = { error: "Network error." };
      } finally {
          uploading = false;
      }
  };

  const toggleScreen = () => {
      showScreenA = !showScreenA;
  };
</script>

<main>
  <h1>Menu Text Detection System Playground</h1>
  <input type="file" accept="image/*" on:change={handleImageChange} />
  <button on:click={handleUpload}>Upload</button><br>
  <button on:click={toggleScreen}>Check Server Response Json</button>

  {#if uploading}
      <p>Uploading... Please wait.</p>
  {:else}
      {#if showScreenA}
          {#if serverResponse}
              {#if serverResponse.error}
                  <p>Error: {serverResponse.error}</p>
              {:else}
                  {#if editedData}
                      <div>
                          <h2>Restaurant Data:</h2>
                            {#if 'restaurant_name' in editedData}
                                <label>餐廳名稱: <input bind:value={editedData.restaurant_name} /></label> <br>
                            {:else}
                                <label>餐廳名稱: <input value="" /></label> <br>
                            {/if}

                            {#if 'business_hours' in editedData}
                                <label>營業時間: <input bind:value={editedData.business_hours} /></label> <br>
                            {:else}
                                <label>營業時間: <input value="" /></label> <br>
                            {/if}

                            {#if 'contact' in editedData}
                                {#if 'address' in editedData.contact}
                                    <label>餐廳地址: <input bind:value={editedData.contact.address} /></label> <br>
                                {:else}
                                    <label>餐廳地址: <input value="" /></label> <br>
                                {/if}
                            {:else}
                                <label>餐廳地址: <input value="" /></label> <br>
                            {/if}

                            {#if 'contact' in editedData}
                                {#if 'phone' in editedData.contact}
                                    <label>聯絡電話: <input bind:value={editedData.contact.phone} /></label> <br>
                                {:else}
                                    <label>聯絡電話: <input value="" /></label> <br>
                                {/if}
                            {:else}
                                <label>聯絡電話: <input value="" /></label> <br>
                            {/if}

            
                          <h3>Dish Information:</h3>
                          {#each editedData.dish as dish, i (dish.name)}
                              <div>
                                  <label>菜名: <input bind:value={editedData.dish[i].name} /></label>
                                  <label>價錢: <input bind:value={editedData.dish[i].price} type="number" style="width: 30px;"/></label>
                              </div>
                          {/each}
                        
                          {#if 'other' in editedData}
                            <label>其他資訊: <textarea bind:value={editedData.other}></textarea></label>
                          {:else}
                            <label>其他資訊: <textarea value=""></textarea></label>
                          {/if}
                      </div>
                  {/if}
              {/if}
          {/if}
      {/if}
      {#if !showScreenA}
          <pre>{JSON.stringify(restaurantData, null, 2)}</pre>
      {/if}
  {/if}
</main>
