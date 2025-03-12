<template>
    <div>
        <div v-if="debug">
            <div>{{key}}={{value}} </div>
        </div>
        <div v-else style="display: none">
        </div>
    </div>
</template>
<script>
module.exports = {
    created() {
        if(this.debug) {
            console.log("localStorage store: created for", this.key, this.value);
        }
        const initialValue = this.readLocalStorage()
        if(initialValue !== null) {
            if(this.debug) {
                console.log("found initial value for localStorage store", this.key, "=", initialValue)
            }
            this.value = initialValue;
        } else {
            if(this.debug) {
                console.log("no initial value for localStorage store", this.key)
            }
            this.writeLocalStorage();
        }
    },
    methods: {
        readLocalStorage() {
            return localStorage.getItem(this.key)
        },
        writeLocalStorage() {
            if(this.debug) {
                if(this.value === null) {
                    console.log("removing key from localStorage");
                } else {
                    console.log("set localStorage value to", this.value);
                }
            }
            let exp = new Date(new Date().setFullYear(new Date().getFullYear() + 10)).toUTCString()
            if(this.value === null) {
                localStorage.removeItem(this.key)
            } else {
                localStorage.setItem(this.key, this.value)
            }
            if(this.storage_written) {
                this.storage_written()
            }
        }

    },
    watch: {
        value(v) {
            this.writeLocalStorage()
        }
    },
}
</script>
